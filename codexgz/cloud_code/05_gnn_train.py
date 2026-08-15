"""
05_gnn_train.py - GNN 图神经网络 Baseline
=========================================
使用 GAT (Graph Attention Network) 做节点级根因分类。
- 输入：完整知识图谱（Alarm + Device 节点）
- 输出：每个 Alarm 节点的根因概率
- 监督信号：仅对 Alarm 节点计算 loss

运行方式：python 05_gnn_train.py
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

# ========== 配置 ==========
TRAIN_DIR = Path("D:/zgyidong/train")
TEST_DIR = Path("D:/zgyidong/test")
MODEL_DIR = Path("D:/zgyidong/code/models")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("D:/zgyidong/code/submit")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001
HIDDEN = 128
HEADS = 4


def stable_hash(s, mod=200):
    if not isinstance(s, str) or not s: return 0
    h = 5381
    for c in s: h = ((h << 5) + h) + ord(c); h &= 0xFFFFFFFF
    return h % mod


def safe_int(v, default=0):
    try: return int(v)
    except: return default


def json_to_pyg(wo_dir, mode="train"):
    """将一个工单的 JSON 转为 PyG Data 对象"""
    topo_path = wo_dir / f"{wo_dir.name}.log.topo.json"
    rc_path = wo_dir / f"{wo_dir.name}.rootcause.json"
    
    obj = json.loads(topo_path.read_text(encoding="utf-8"))
    nodes = obj["nodes"]
    edges = obj.get("edges", [])
    ft = safe_int(obj.get("time", 0))
    
    n = len(nodes)
    rid2idx = {nd["@rid"]: i for i, nd in enumerate(nodes)}
    
    # === 节点特征 (embedding) ===
    x_list = []
    alarm_mask = []
    
    for i, nd in enumerate(nodes):
        cls = nd.get("@class", "Unknown")
        is_alarm = 1 if cls == "Alarm" else 0
        alarm_mask.append(is_alarm)
        
        f = []
        # 基础
        f.append(is_alarm)
        f.append(1 if nd.get("label") == "TargetAlarm" else 0)
        f.append(stable_hash(cls, 50))
        
        if is_alarm:
            f.append(stable_hash(nd.get("title", ""), 80))
            f.append(stable_hash(nd.get("vendor", ""), 10))
            f.append(stable_hash(nd.get("device", ""), 80))
            t = safe_int(nd.get("time", ft))
            f.append(min(max(0, (ft - t) / 60000), 1440))
            
            tl = nd.get("timeLists", [])
            if isinstance(tl, list) and len(tl) >= 6:
                tla = np.array(tl[:6], dtype=np.float32)
                f.append(float(np.sum(tla)))
                f.append(float(np.mean(tla)))
                f.append(float(tla[-1]))
            else:
                f.extend([0, 0, 0])
            
            reason = nd.get("reason", "")
            f.append(1 if "市电" in reason else 0)
            f.append(1 if "故障" in reason else 0)
            f.append(len(reason) // 20)
            f.extend([0] * 6)  # pad to 19
        else:
            f.extend([0] * 16)  # 16 zeros for non-alarm to match dim 19
        
        x_list.append(f)
    
    x = torch.tensor(x_list, dtype=torch.float32)
    
    # === 边索引 ===
    edge_list = []
    for e in edges:
        s = e.get("in", "")
        t = e.get("out", "")
        if s in rid2idx and t in rid2idx:
            edge_list.append([rid2idx[s], rid2idx[t]])
            edge_list.append([rid2idx[t], rid2idx[s]])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # === 标签 (仅 Alarm 节点) ===
    alarm_mask_t = torch.tensor(alarm_mask, dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.long)
    
    if mode == "train" and rc_path.exists():
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        rc_rids = set(r["@rid"] for r in rc.get("rootcause", []))
        for i, nd in enumerate(nodes):
            if nd["@rid"] in rc_rids and alarm_mask[i]:
                y[i] = 1
    
    return Data(x=x, edge_index=edge_index, y=y, alarm_mask=alarm_mask_t)


class RootCauseGNN(nn.Module):
    """GAT-based root cause classifier"""
    def __init__(self, in_dim, hidden=HIDDEN, heads=HEADS, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        
        # 节点级分类
        logits = self.classifier(x).squeeze(-1)
        return logits


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        logits = model(data)
        
        # 只在 Alarm 节点上计算 loss
        mask = data.alarm_mask
        loss = criterion(logits[mask], data.y[mask].float())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data)
        probs = torch.sigmoid(logits)
        
        mask = data.alarm_mask
        preds = (probs[mask] >= 0.5).long()
        all_preds.append(preds.cpu())
        all_labels.append(data.y[mask].cpu())
    
    yp = torch.cat(all_preds).numpy()
    yt = torch.cat(all_labels).numpy()
    
    return {
        "f1": f1_score(yt, yp, zero_division=0),
        "prec": precision_score(yt, yp, zero_division=0),
        "rec": recall_score(yt, yp, zero_division=0),
    }


@torch.no_grad()
def predict_test(model, test_dirs):
    """对测试集预测"""
    model.eval()
    all_preds = {}
    k_vals = []
    
    for wo in test_dirs:
        try:
            data = json_to_pyg(wo, mode="test")
            data = data.to(DEVICE)
            
            logits = model(data)
            probs = torch.sigmoid(logits)
            
            # 只关注 Alarm 节点
            mask = data.alarm_mask
            alarm_probs = probs[mask].detach().cpu().numpy()
            
            # 找到对应的节点信息
            topo_path = wo / f"{wo.name}.log.topo.json"
            topo = json.loads(topo_path.read_text(encoding="utf-8"))
            nodes = topo["nodes"]
            alarm_nodes = [n for n in nodes if n.get("@class") == "Alarm"]
            
            # 筛选高概率节点
            confident = np.where(alarm_probs >= 0.5)[0]
            if len(confident) == 0:
                k = 1
            elif len(confident) > 5:
                k = 5
            else:
                k = len(confident)
            
            ranked = sorted(range(len(alarm_probs)), key=lambda i: alarm_probs[i], reverse=True)
            top_k = ranked[:k]
            k_vals.append(len(top_k))
            
            rc_list = []
            for idx in top_k:
                nd = alarm_nodes[idx]
                rc_list.append({
                    "@rid": nd["@rid"],
                    "title": nd.get("title", ""),
                    "location": nd.get("location", ""),
                    "reason": nd.get("reason", ""),
                })
            
            all_preds[wo.name] = {"rootcause": rc_list}
        except Exception as e:
            print(f"  ⚠ {wo.name}: {e}")
    
    return all_preds, k_vals


def main():
    print(f"Device: {DEVICE}")
    
    # 1. 加载数据
    print("\n加载工单...")
    train_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])[:1000]  # 前1000训练
    val_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])[1500:1550]  # 50个验证
    test_dirs = sorted([d for d in TEST_DIR.iterdir() if d.is_dir()])
    
    print(f"Train: {len(train_dirs)}, Val: {len(val_dirs)}, Test: {len(test_dirs)}")
    
    # 构建 Data 列表
    print("构建图数据...")
    train_data = [json_to_pyg(d, "train") for d in train_dirs if (d / f"{d.name}.log.topo.json").exists()]
    val_data = [json_to_pyg(d, "train") for d in val_dirs if (d / f"{d.name}.log.topo.json").exists()]
    
    train_data = [d for d in train_data if d.alarm_mask.sum() > 0]
    val_data = [d for d in val_data if d.alarm_mask.sum() > 0]
    
    print(f"有效训练图: {len(train_data)}, 验证图: {len(val_data)}")
    
    # 2. 模型
    in_dim = train_data[0].x.shape[1]
    model = RootCauseGNN(in_dim).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # 加权 BCELoss
    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)  # batch=1 因为图大小不同
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    # 3. 训练
    print(f"\n开始训练 ({EPOCHS} epochs)...")
    best_f1 = 0
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, val_loader)
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, F1={metrics['f1']:.4f}, Prec={metrics['prec']:.4f}, Rec={metrics['rec']:.4f}")
            
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(model.state_dict(), MODEL_DIR / "gnn_best.pt")
    
    print(f"\nBest val F1: {best_f1:.4f}")
    
    # 4. 预测测试集
    print("\n加载最佳模型预测测试集...")
    model.load_state_dict(torch.load(MODEL_DIR / "gnn_best.pt", weights_only=True))
    
    preds, k_vals = predict_test(model, test_dirs)
    
    out_path = OUTPUT_DIR / "predictions_gnn.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    
    print(f"✅ GNN 预测完成!")
    print(f"   工单: {len(preds)}, 平均根因数: {np.mean(k_vals):.1f}")
    print(f"   文件: {out_path}")


if __name__ == "__main__":
    main()
