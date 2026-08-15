"""Train one honest V24 outer fold and predict its validation/test rows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from v24_data import GraphStore, batch_graphs, shuffled_batches
from v24_model import V24HeteroPathRanker, multitask_loss


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
    if int(selected.sum()) != target_count:
        raise ValueError((int(selected.sum()), target_count))
    return selected


def fixed_k_tp(scores, store, order_indices):
    total = 0
    for order_index in order_indices:
        start = int(store.alarm_ptr[order_index])
        stop = int(store.alarm_ptr[order_index + 1])
        count = int(store.base_mask[start:stop].sum())
        ranked = np.argsort(-scores[start:stop], kind="stable")[:count]
        total += int(store.labels[start:stop][ranked].sum())
    return total


def make_model(config, store, device):
    return V24HeteroPathRanker(
        alarm_feature_dim=store.alarm_x.shape[1],
        node_numeric_dim=store.node_numeric.shape[1],
        hidden_dim=config["hidden_dim"],
        heads=config["heads"],
        graph_layers=config["graph_layers"],
        path_layers=config["path_layers"],
        node_types=len(store.metadata["node_classes"]),
        relations=store.metadata["relation_count"],
        max_path_nodes=store.metadata["max_path_nodes"],
        dropout=config["dropout"],
    ).to(device)


def make_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


def train_epoch(model, optimizer, store, indices, config, positive_weight, device, rng):
    model.train()
    totals = []
    parts = []
    for batch_indices in shuffled_batches(indices, config["batch_size"], rng):
        batch = batch_graphs(store, batch_indices, device)
        output = model(batch)
        loss, metrics = multitask_loss(
            output,
            batch,
            config["loss_weights"],
            positive_weight,
            config["hard_negatives_per_positive"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals.append(float(loss.detach().cpu()))
        parts.append(metrics)
    summary = {name: float(np.mean([item[name] for item in parts])) for name in parts[0]}
    return float(np.mean(totals)), summary


@torch.no_grad()
def predict(model, store, order_indices, batch_size, device):
    model.eval()
    scores = np.zeros(len(store.v11), dtype=np.float32)
    gates = np.zeros(len(store.v11), dtype=np.float32)
    count_probs = np.zeros((len(store), 8), dtype=np.float32)
    for batch_start in range(0, len(order_indices), batch_size):
        indices = np.asarray(order_indices[batch_start : batch_start + batch_size], dtype=np.int64)
        batch = batch_graphs(store, indices, device)
        output = model(batch)
        cursor = 0
        probabilities = torch.softmax(output["count_logits"], dim=-1).float().cpu().numpy()
        local_scores = output["logits"].float().cpu().numpy()
        local_gates = output["gate"].float().cpu().numpy()
        for local_order, order_index in enumerate(indices):
            start = int(store.alarm_ptr[order_index])
            stop = int(store.alarm_ptr[order_index + 1])
            length = stop - start
            scores[start:stop] = local_scores[cursor : cursor + length]
            gates[start:stop] = local_gates[cursor : cursor + length]
            count_probs[order_index] = probabilities[local_order]
            cursor += length
    return scores, gates, count_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_probe.json")
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
    target_train = round(config["target_test_predictions"] / len(test) * len(train))
    train.base_mask = exact_count_mask(
        train.v11, train.alarm_ptr, target_train, config["max_rootcauses"]
    )
    test.base_mask = test.champion_mask

    outer_train = np.flatnonzero(train.folds != args.fold)
    validation = np.flatnonzero(train.folds == args.fold)
    seed_index = config["seeds"].index(args.seed)
    monitor_fold = (args.fold + 1 + seed_index) % config["folds"]
    monitor = np.flatnonzero(train.folds == monitor_fold)
    fit = np.flatnonzero((train.folds != args.fold) & (train.folds != monitor_fold))

    model = make_model(config, train, device)
    optimizer = make_optimizer(model, config)
    positive_weight = torch.tensor(config["positive_weight"], dtype=torch.float32, device=device)
    monitor_base_tp = fixed_k_tp(train.v11, train, monitor)
    best = {"tp_delta": -10**9, "loss": float("inf"), "epoch": -1}
    stale = 0
    rng = np.random.default_rng(args.seed + args.fold)
    for epoch in range(config["epochs"]):
        loss, parts = train_epoch(
            model, optimizer, train, fit, config, positive_weight, device, rng
        )
        monitor_scores, _, _ = predict(
            model, train, monitor, config["predict_batch_size"], device
        )
        delta = fixed_k_tp(monitor_scores, train, monitor) - monitor_base_tp
        improved = delta > best["tp_delta"] or (delta == best["tp_delta"] and loss < best["loss"])
        if improved:
            best = {"tp_delta": int(delta), "loss": loss, "epoch": epoch, "loss_parts": parts}
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "loss": loss, "parts": parts, "monitor_tp_delta": int(delta)}), flush=True)
        if stale >= config["patience"]:
            break

    selected_epochs = best["epoch"] + 1
    seed_everything(args.seed + args.fold)
    model = make_model(config, train, device)
    optimizer = make_optimizer(model, config)
    rng = np.random.default_rng(args.seed + args.fold)
    refit_loss = None
    for epoch in range(selected_epochs):
        refit_loss, _ = train_epoch(
            model, optimizer, train, outer_train, config, positive_weight, device, rng
        )
        print(json.dumps({"refit_epoch": epoch, "loss": refit_loss}), flush=True)

    validation_scores, validation_gates, validation_counts = predict(
        model, train, validation, config["predict_batch_size"], device
    )
    test_scores, test_gates, test_counts = predict(
        model, test, np.arange(len(test)), config["predict_batch_size"], device
    )
    validation_rows = np.concatenate(
        [
            np.arange(int(train.alarm_ptr[index]), int(train.alarm_ptr[index + 1]))
            for index in validation
        ]
    )
    output_path = args.output / f"seed_{args.seed}_fold_{args.fold}.npz"
    np.savez_compressed(
        output_path,
        validation_order_indices=validation,
        validation_rows=validation_rows,
        validation_scores=validation_scores[validation_rows],
        validation_gates=validation_gates[validation_rows],
        validation_count_probs=validation_counts[validation],
        test_scores=test_scores,
        test_gates=test_gates,
        test_count_probs=test_counts,
    )
    checkpoint = args.output / f"seed_{args.seed}_fold_{args.fold}.pt"
    torch.save({"model": model.state_dict(), "config": config, "best": best}, checkpoint)
    report = {
        "device": str(device),
        "fold": args.fold,
        "seed": args.seed,
        "monitor_fold": int(monitor_fold),
        "selected_epochs": int(selected_epochs),
        "monitor_best": best,
        "outer_refit_loss": refit_loss,
        "output": str(output_path),
    }
    (args.output / f"seed_{args.seed}_fold_{args.fold}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
