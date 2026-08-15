"""Train one honest V25 semantic fold and persist OOF/test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semantic_model import (
    last_token_embedding,
    load_qwen,
    make_set_ranker,
    multitask_loss,
    order_prompts,
    resolve_device,
)
from v25_common import Bundle, exact_count_mask, read_json, seed_everything, write_json


def numeric_features(bundle: Bundle, split: str) -> np.ndarray:
    arrays = bundle.arrays
    experts = np.column_stack(
        [
            arrays[f"{split}_v11"],
            np.nan_to_num(arrays[f"{split}_v13"], nan=arrays[f"{split}_v11"]),
            np.nan_to_num(arrays[f"{split}_v19"], nan=arrays[f"{split}_v11"]),
            np.minimum(arrays[f"{split}_path_length"], 12) / 12.0,
        ]
    )
    return np.concatenate([arrays[f"{split}_alarm_x"], experts], axis=1).astype(np.float32)


def candidate_rows(bundle, split, order_indices):
    ptr = bundle.ptr(split)
    records = bundle.records[split]
    v11 = bundle.arrays[f"{split}_v11"]
    for order_index in order_indices:
        start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
        prompts = order_prompts(records[order_index]["alarms"], v11[start:stop])
        for local, prompt in enumerate(prompts):
            yield start + local, prompt


def train_candidate_lora(bundle, model_path, dapt, config, train_orders, epochs, device):
    import torch

    tokenizer, model = load_qwen(model_path, config, dapt_adapter=dapt, trainable=True)
    model.to(device).train()
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(value for value in model.parameters() if value.requires_grad) + list(head.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    labels = bundle.arrays["train_labels"]
    steps = 0
    rows = list(candidate_rows(bundle, "train", train_orders))
    for _ in range(epochs):
        np.random.shuffle(rows)
        for row, prompt in rows:
            embedding = last_token_embedding(
                model, tokenizer, [prompt], device, config["max_length"], gradients=True
            )
            logit = head(embedding).squeeze()
            target = torch.tensor(float(labels[row]), device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
            (loss / config["gradient_accumulation"]).backward()
            steps += 1
            if steps % config["gradient_accumulation"] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if steps % config["gradient_accumulation"]:
        optimizer.step()
    return tokenizer, model, head


def select_candidate_epochs(bundle, model_path, dapt, config, train_orders, validation_orders, device):
    import torch

    tokenizer, model = load_qwen(model_path, config, dapt_adapter=dapt, trainable=True)
    model.to(device).train()
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(value for value in model.parameters() if value.requires_grad) + list(head.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    labels = bundle.arrays["train_labels"]
    rows = list(candidate_rows(bundle, "train", train_orders))
    best_epoch, best_bce, steps = 1, float("inf"), 0
    for epoch in range(1, config["epochs"] + 1):
        np.random.shuffle(rows)
        model.train()
        for row, prompt in rows:
            embedding = last_token_embedding(
                model, tokenizer, [prompt], device, config["max_length"], gradients=True
            )
            logit = head(embedding).squeeze()
            target = torch.tensor(float(labels[row]), device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
            (loss / config["gradient_accumulation"]).backward()
            steps += 1
            if steps % config["gradient_accumulation"] == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if steps % config["gradient_accumulation"]:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        value = candidate_bce(
            bundle, tokenizer, model, head, validation_orders, device, config
        )
        if value < best_bce:
            best_epoch, best_bce = epoch, value
    return best_epoch, best_bce


def candidate_bce(bundle, tokenizer, model, head, order_indices, device, config):
    import torch

    losses = []
    labels = bundle.arrays["train_labels"]
    model.eval()
    for row, prompt in candidate_rows(bundle, "train", order_indices):
        embedding = last_token_embedding(model, tokenizer, [prompt], device, config["max_length"])
        logit = head(embedding).squeeze()
        target = torch.tensor(float(labels[row]), device=device)
        losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logit, target).cpu()))
    return float(np.mean(losses))


def encode_split(bundle, split, tokenizer, model, device, config):
    model.eval()
    output = None
    for order_index in range(len(bundle.records[split])):
        ptr = bundle.ptr(split)
        start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
        prompts = order_prompts(
            bundle.records[split][order_index]["alarms"],
            bundle.arrays[f"{split}_v11"][start:stop],
        )
        chunks = []
        for offset in range(0, len(prompts), 2):
            chunks.append(
                last_token_embedding(
                    model, tokenizer, prompts[offset : offset + 2], device, config["max_length"]
                ).cpu().numpy().astype(np.float16)
            )
        values = np.concatenate(chunks)
        if output is None:
            output = np.zeros((int(ptr[-1]), values.shape[1]), dtype=np.float16)
        output[start:stop] = values
    return output


def train_set_ranker(bundle, embeddings, fold, inner_fold, config, seed, device):
    import torch

    ptr = bundle.ptr("train")
    folds = bundle.arrays["train_folds"]
    labels = bundle.arrays["train_labels"]
    v11 = bundle.arrays["train_v11"]
    numeric = numeric_features(bundle, "train")
    base = exact_count_mask(v11, ptr, 3169)
    model = make_set_ranker(embeddings.shape[1], numeric.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_orders = np.flatnonzero((folds != fold) & (folds != inner_fold))
    validation = np.flatnonzero(folds == inner_fold)
    best_epoch, best_loss, stale = 1, float("inf"), 0
    for epoch in range(60):
        np.random.shuffle(train_orders)
        model.train()
        for order_index in train_orders:
            start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
            output = model(
                torch.as_tensor(embeddings[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(numeric[start:stop], device=device),
            )
            loss, _ = multitask_loss(
                output,
                torch.as_tensor(labels[start:stop], device=device),
                torch.as_tensor(base[start:stop], device=device),
                int(labels[start:stop].sum()),
                config["loss_weights"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses = []
        model.eval()
        with torch.no_grad():
            for order_index in validation:
                start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
                output = model(
                    torch.as_tensor(embeddings[start:stop], dtype=torch.float32, device=device),
                    torch.as_tensor(numeric[start:stop], device=device),
                )
                loss, _ = multitask_loss(
                    output,
                    torch.as_tensor(labels[start:stop], device=device),
                    torch.as_tensor(base[start:stop], device=device),
                    int(labels[start:stop].sum()),
                    config["loss_weights"],
                )
                losses.append(float(loss.cpu()))
        value = float(np.mean(losses))
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= 6:
            break
    # Refit from a fresh initialization on every non-outer order. The outer
    # fold was not used for epoch selection or parameter fitting.
    seed_everything(seed + 10007)
    model = make_set_ranker(embeddings.shape[1], numeric.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_orders = np.flatnonzero(folds != fold)
    for _ in range(best_epoch):
        np.random.shuffle(train_orders)
        model.train()
        for order_index in train_orders:
            start, stop = int(ptr[order_index]), int(ptr[order_index + 1])
            output = model(
                torch.as_tensor(embeddings[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(numeric[start:stop], device=device),
            )
            loss, _ = multitask_loss(
                output,
                torch.as_tensor(labels[start:stop], device=device),
                torch.as_tensor(base[start:stop], device=device),
                int(labels[start:stop].sum()),
                config["loss_weights"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, best_loss, best_epoch


def predict_set(bundle, split, embeddings, model, device):
    import torch

    ptr = bundle.ptr(split)
    numeric = numeric_features(bundle, split)
    scores = np.zeros(int(ptr[-1]), dtype=np.float32)
    counts = np.zeros((len(ptr) - 1, 8), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
            start, stop = int(start), int(stop)
            output = model(
                torch.as_tensor(embeddings[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(numeric[start:stop], device=device),
            )
            scores[start:stop] = torch.sigmoid(output["logits"]).cpu().numpy()
            counts[index] = torch.softmax(output["count_logits"], dim=0).cpu().numpy()
    return scores, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dapt-adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = read_json(args.config)
    model_config = config["model"]
    seed_everything(args.seed)
    device = resolve_device(args.device)
    bundle = Bundle.load(args.data_root)
    folds = bundle.arrays["train_folds"]
    inner_fold = (args.fold + 1) % config["folds"]
    trial_orders = np.flatnonzero((folds != args.fold) & (folds != inner_fold))
    inner_orders = np.flatnonzero(folds == inner_fold)
    best_epoch, best_bce = select_candidate_epochs(
        bundle, args.model, args.dapt_adapter, model_config, trial_orders, inner_orders, device
    )
    try:
        import torch
        if device.type == "npu":
            torch.npu.empty_cache()
    except Exception:
        pass
    outer_train = np.flatnonzero(folds != args.fold)
    tokenizer, model, _ = train_candidate_lora(
        bundle, args.model, args.dapt_adapter, model_config, outer_train, best_epoch, device
    )
    train_embeddings = encode_split(bundle, "train", tokenizer, model, device, model_config)
    test_embeddings = encode_split(bundle, "test", tokenizer, model, device, model_config)
    set_model, set_loss, set_epochs = train_set_ranker(
        bundle, train_embeddings, args.fold, inner_fold, model_config, args.seed, device
    )
    train_scores, train_counts = predict_set(bundle, "train", train_embeddings, set_model, device)
    test_scores, test_counts = predict_set(bundle, "test", test_embeddings, set_model, device)
    validation_orders = np.flatnonzero(folds == args.fold)
    validation_rows = bundle.rows_for_orders("train", validation_orders)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / f"semantic_seed_{args.seed}_fold_{args.fold}.npz",
        validation_orders=validation_orders,
        validation_rows=validation_rows,
        validation_scores=train_scores[validation_rows],
        validation_counts=train_counts[validation_orders],
        test_scores=test_scores,
        test_counts=test_counts,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
    )
    write_json(
        args.output / f"semantic_seed_{args.seed}_fold_{args.fold}.json",
        {
            "seed": args.seed,
            "fold": args.fold,
            "inner_fold": inner_fold,
            "selected_lora_epochs": best_epoch,
            "inner_bce": best_bce,
            "set_validation_loss": set_loss,
            "selected_set_epochs": set_epochs,
            "device": str(device),
        },
    )


if __name__ == "__main__":
    main()
