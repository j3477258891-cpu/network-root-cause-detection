"""Train one grouped OOF fold/seed and predict the test set."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from v17_data import GraphStore, batch_graphs, shuffled_batches
from v17_model import ResidualRGT, pairwise_loss


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass


def exact_count_mask(scores, ptr, target_count, max_roots=8):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for index in range(len(ptr) - 1):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        ranked = np.argsort(-scores[start:stop], kind="stable")[:max_roots]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    remaining = target_count - int(selected.sum())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def fixed_k_tp(scores, store, order_indices, base_mask):
    tp = 0
    for order_index in order_indices:
        start = int(store.order_ptr[order_index])
        stop = int(store.order_ptr[order_index + 1])
        count = int(base_mask[start:stop].sum())
        ranked = np.argsort(-scores[start:stop], kind="stable")[:count]
        tp += int(store.labels[start:stop][ranked].sum())
    return tp


@torch.no_grad()
def predict(model, store, order_indices, batch_size, device):
    model.eval()
    node_scores = np.zeros(len(store.x), dtype=np.float32)
    count_probs = np.zeros((len(store), 8), dtype=np.float32)
    gates = np.zeros(len(store.x), dtype=np.float32)
    for start in range(0, len(order_indices), batch_size):
        indices = np.asarray(order_indices[start : start + batch_size], dtype=np.int64)
        batch = batch_graphs(store, indices, device)
        output = model(batch)
        local_cursor = 0
        for local_graph, order_index in enumerate(indices):
            row_start = int(store.order_ptr[order_index])
            row_stop = int(store.order_ptr[order_index + 1])
            length = row_stop - row_start
            node_scores[row_start:row_stop] = output["logits"][
                local_cursor : local_cursor + length
            ].float().cpu().numpy()
            gates[row_start:row_stop] = output["gate"][
                local_cursor : local_cursor + length
            ].float().cpu().numpy()
            count_probs[order_index] = F.softmax(output["count_logits"][local_graph], dim=-1).float().cpu().numpy()
            local_cursor += length
    return node_scores, count_probs, gates


def make_model_optimizer(config, feature_dim, device):
    model = ResidualRGT(
        feature_dim=feature_dim,
        hidden_dim=config["hidden_dim"],
        heads=config["heads"],
        layers=config["layers"],
        relation_count=5,
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    return model, optimizer


def train_epoch(model, optimizer, store, indices, config, pos_weight, device, rng):
    model.train()
    losses = []
    for batch_indices in shuffled_batches(indices, config["batch_size"], rng):
        batch = batch_graphs(store, batch_indices, device)
        output = model(batch)
        bce = F.binary_cross_entropy_with_logits(
            output["logits"], batch["labels"], pos_weight=pos_weight
        )
        pair = pairwise_loss(
            output["logits"],
            batch["labels"],
            batch["v11"],
            batch["ptr"],
            config["hard_negatives_per_positive"],
        )
        count = F.cross_entropy(output["count_logits"], batch["counts"])
        weights = config["loss_weights"]
        loss = weights["bce"] * bce + weights["pairwise"] * pair + weights["count"] * count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed_everything(args.seed + args.fold)
    device = resolve_device(args.device)
    train = GraphStore(args.data_root, "train")
    test = GraphStore(args.data_root, "test")
    outer_train_indices = np.flatnonzero(train.folds != args.fold)
    validation_indices = np.flatnonzero(train.folds == args.fold)
    honest_outer_cv = bool(config.get("honest_outer_cv", False))
    if honest_outer_cv:
        seed_index = config["seeds"].index(args.seed)
        monitor_fold = (args.fold + 1 + seed_index) % config["folds"]
        monitor_indices = np.flatnonzero(train.folds == monitor_fold)
        train_indices = np.flatnonzero(
            (train.folds != args.fold) & (train.folds != monitor_fold)
        )
    else:
        monitor_fold = args.fold
        monitor_indices = validation_indices
        train_indices = outer_train_indices
    target_count = round(config["target_test_predictions"] / len(test) * len(train))
    base_mask = exact_count_mask(train.v11, train.order_ptr, target_count, config["max_rootcauses"])
    monitor_base_tp = sum(
        int(train.labels[int(train.order_ptr[i]) : int(train.order_ptr[i + 1])][base_mask[int(train.order_ptr[i]) : int(train.order_ptr[i + 1])]].sum())
        for i in monitor_indices
    )
    model, optimizer = make_model_optimizer(config, train.x.shape[1], device)
    pos_weight = torch.tensor(config["positive_weight"], dtype=torch.float32, device=device)
    best = {"tp_delta": -10**9, "loss": float("inf"), "epoch": -1, "state": None}
    stale = 0
    rng = np.random.default_rng(args.seed + args.fold)
    for epoch in range(config["epochs"]):
        mean_loss = train_epoch(
            model, optimizer, train, train_indices, config, pos_weight, device, rng
        )
        monitor_scores, _, _ = predict(
            model, train, monitor_indices, config["batch_size"], device
        )
        candidate_tp = fixed_k_tp(monitor_scores, train, monitor_indices, base_mask)
        delta = candidate_tp - monitor_base_tp
        improved = delta > best["tp_delta"] or (delta == best["tp_delta"] and mean_loss < best["loss"])
        if improved:
            best = {
                "tp_delta": int(delta),
                "loss": mean_loss,
                "epoch": epoch,
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "loss": mean_loss, "fixed_k_tp_delta": delta}), flush=True)
        if stale >= config["patience"]:
            break
    if honest_outer_cv:
        selected_epochs = best["epoch"] + 1
        seed_everything(args.seed + args.fold)
        model, optimizer = make_model_optimizer(config, train.x.shape[1], device)
        rng = np.random.default_rng(args.seed + args.fold)
        refit_losses = []
        for _ in range(selected_epochs):
            refit_losses.append(
                train_epoch(
                    model,
                    optimizer,
                    train,
                    outer_train_indices,
                    config,
                    pos_weight,
                    device,
                    rng,
                )
            )
        best["monitor_fold"] = int(monitor_fold)
        best["outer_refit_epochs"] = int(selected_epochs)
        best["outer_refit_loss"] = float(refit_losses[-1])
        best.pop("state")
    else:
        model.load_state_dict(best.pop("state"))
    validation_scores, validation_counts, validation_gates = predict(
        model, train, validation_indices, config["batch_size"], device
    )
    test_scores, test_counts, test_gates = predict(
        model, test, np.arange(len(test)), config["batch_size"], device
    )
    validation_rows = np.concatenate(
        [
            np.arange(int(train.order_ptr[index]), int(train.order_ptr[index + 1]))
            for index in validation_indices
        ]
    )
    output_path = args.output / f"seed_{args.seed}_fold_{args.fold}.npz"
    np.savez_compressed(
        output_path,
        validation_order_indices=validation_indices,
        validation_rows=validation_rows,
        validation_scores=validation_scores[validation_rows],
        validation_count_probs=validation_counts[validation_indices],
        validation_gates=validation_gates[validation_rows],
        test_scores=test_scores,
        test_count_probs=test_counts,
        test_gates=test_gates,
    )
    checkpoint = args.output / f"seed_{args.seed}_fold_{args.fold}.pt"
    torch.save({"model": model.state_dict(), "config": config, "best": best}, checkpoint)
    (args.output / f"seed_{args.seed}_fold_{args.fold}.json").write_text(
        json.dumps({"device": str(device), "best": best}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), "best": best}), flush=True)


if __name__ == "__main__":
    main()
