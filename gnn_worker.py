import os, gc, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv

# ── 路径配置（与主notebook保持一致）──────────────────────────
BASE_DIR = "./"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ★ 修复1：MODEL_DIRS 和 EXCEL_PATHS 加入 TopACT，去掉 MLP
MODEL_DIRS = {
    "GCN": os.path.join(MODELS_DIR, "GCN"),
    "GraphSAGE": os.path.join(MODELS_DIR, "GraphSAGE"),
    "GAT": os.path.join(MODELS_DIR, "GAT"),
    "GATv2": os.path.join(MODELS_DIR, "GATv2"),
    "TopACT": os.path.join(MODELS_DIR, "TopACT"),
}
EXCEL_PATHS = {
    "GCN": os.path.join(MODEL_DIRS["GCN"], "GCN_results.xlsx"),
    "GraphSAGE": os.path.join(MODEL_DIRS["GraphSAGE"], "GraphSAGE_results.xlsx"),
    "GAT": os.path.join(MODEL_DIRS["GAT"], "GAT_results.xlsx"),
    "GATv2": os.path.join(MODEL_DIRS["GATv2"], "GATv2_results.xlsx"),
    "TopACT": os.path.join(MODEL_DIRS["TopACT"], "TopACT_results.xlsx"),
}

# ── 模型定义（去掉 MLPWithDropout）────────────────────────────


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.skip_proj = nn.Linear(in_dim, hidden_dim) if num_layers >= 2 else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        identity = x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == 0 and self.skip_proj is not None:
                x = x + self.skip_proj(identity)
        return F.log_softmax(self.convs[-1](x, edge_index), dim=1)


class GraphSAGEWithNorm(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.convs[-1](x, edge_index), dim=1)


class GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        heads=8,
        dropout=0.3,
        num_layers=2,
        use_skip=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_skip = use_skip
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        self.convs.append(
            GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)
        )
        if use_skip and num_layers >= 2:
            self.skip_proj = nn.Linear(in_dim, hidden_dim * heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        identity = (
            self.skip_proj(x) if (self.use_skip and self.num_layers >= 2) else None
        )
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if identity is not None and i == 0:
                x = x + identity
        return F.log_softmax(self.convs[-1](x, edge_index), dim=1)


class GATv2Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, dropout=0.3, num_layers=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        self.convs.append(
            GATv2Conv(
                hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False
            )
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.convs[-1](x, edge_index), dim=1)


# ── 训练函数 ──────────────────────────────────────────────────


def clear_memory(device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()


def train_model(model, data, device, optimizer, epochs=500, patience=50, verbose=False):
    data = data.to(device)
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            train_acc = (
                (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            )
        train_losses.append(loss.item())
        train_accs.append(train_acc)

        model.eval()
        with torch.no_grad():
            out_val = model(data)
            val_loss = F.nll_loss(out_val[data.val_mask], data.y[data.val_mask])
            pred_val = out_val.argmax(dim=1)
            val_acc = (
                (pred_val[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            )
        val_losses.append(val_loss.item())
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if use_amp and (epoch + 1) % 100 == 0:
            clear_memory(device)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accs, val_accs, best_val_acc


def evaluate_model(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        test_pred = pred[data.test_mask].cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
    precision, recall, _, _ = precision_recall_fscore_support(
        test_true, test_pred, average="macro"
    )
    return {
        "test_acc": float(accuracy_score(test_true, test_pred)),
        "test_f1_macro": float(f1_score(test_true, test_pred, average="macro")),
        "test_f1_weighted": float(f1_score(test_true, test_pred, average="weighted")),
        "test_precision_macro": float(precision),
        "test_recall_macro": float(recall),
        "test_pred": test_pred.tolist(),
        "test_true": test_true.tolist(),
    }


def save_model_and_results(model, config, model_name, training_history=None):
    # ★ 修复2：动态构建路径，不再依赖模块级固定字典（支持运行时传入的 MODEL_DIRS/EXCEL_PATHS）
    model_dir = MODEL_DIRS.get(model_name)
    excel_path = EXCEL_PATHS.get(model_name)
    if model_dir is None or excel_path is None:
        raise KeyError(
            f"model_name '{model_name}' 不在 MODEL_DIRS/EXCEL_PATHS 中，请检查 gnn_worker.py 顶部的字典配置。"
        )

    os.makedirs(model_dir, exist_ok=True)

    model_filename = f"{model_name}_h{config['hidden_dim']}_d{config['dropout']}_l{config['num_layers']}"
    if "heads" in config:
        model_filename += f"_heads{config['heads']}"
    if "use_skip" in config:
        model_filename += f"_skip{config['use_skip']}"
    model_filename += ".pt"
    model_path = os.path.join(model_dir, model_filename)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_val_acc": config["best_val_acc"],
        "test_acc": config["test_acc"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if training_history:
        save_dict["training_history"] = training_history
    torch.save(save_dict, model_path)

    result_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hidden_dim": config["hidden_dim"],
        "dropout": config["dropout"],
        "num_layers": config["num_layers"],
        "best_val_acc": config["best_val_acc"],
        "test_acc": config["test_acc"],
        "test_f1_macro": config["test_f1_macro"],
        "test_f1_weighted": config["test_f1_weighted"],
        "test_precision_macro": config.get("test_precision_macro", 0),
        "test_recall_macro": config.get("test_recall_macro", 0),
        "num_params": config.get("num_params", 0),
        "model_file": model_filename,
    }
    if "heads" in config:
        result_row["heads"] = config["heads"]
    if "use_skip" in config:
        result_row["use_skip"] = config["use_skip"]

    import time, random

    for attempt in range(10):
        try:
            if os.path.exists(excel_path):
                existing_df = pd.read_excel(excel_path)
                results_df = pd.concat(
                    [existing_df, pd.DataFrame([result_row])], ignore_index=True
                )
            else:
                results_df = pd.DataFrame([result_row])
            results_df = results_df.sort_values("best_val_acc", ascending=False)
            results_df.to_excel(excel_path, index=False)
            break
        except Exception:
            time.sleep(random.uniform(0.5, 2.0))


def save_detailed_results(
    model_name, config, training_history, test_true, test_pred, class_names_list
):
    from sklearn.metrics import precision_recall_fscore_support as prfs

    precision, recall, f1, support = prfs(test_true, test_pred, average=None)
    param_str = f"h{config['hidden_dim']}_d{config['dropout']}_l{config['num_layers']}"
    if "heads" in config:
        param_str += f"_heads{config['heads']}"

    model_dir = MODEL_DIRS.get(model_name, os.path.join(MODELS_DIR, model_name))
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, f"{model_name}_{param_str}.json")
    detailed = {
        "model_name": model_name,
        "config": {
            k: v for k, v in config.items() if k not in ["test_true", "test_pred"]
        },
        "best_val_acc": float(config["best_val_acc"]),
        "test_metrics": {
            "accuracy": float(config["test_acc"]),
            "f1_macro": float(config["test_f1_macro"]),
            "f1_weighted": float(config["test_f1_weighted"]),
            "precision_macro": float(config.get("test_precision_macro", 0)),
            "recall_macro": float(config.get("test_recall_macro", 0)),
        },
        "per_class_metrics": {
            f"class_{i}": {
                "class_name": str(class_names_list[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names_list))
        },
    }
    with open(save_path, "w") as fp:
        json.dump(detailed, fp, indent=2)


def build_model(model_name, params, in_dim, out_dim):
    # ★ MLP 已去除；TopACT 不走此函数（由主进程处理）
    if model_name == "GCN":
        return GCN(
            in_dim,
            params["hidden_dim"],
            out_dim,
            params["dropout"],
            params["num_layers"],
        )
    elif model_name == "GraphSAGE":
        return GraphSAGEWithNorm(
            in_dim,
            params["hidden_dim"],
            out_dim,
            params["dropout"],
            params["num_layers"],
        )
    elif model_name == "GAT":
        return GAT(
            in_dim,
            params["hidden_dim"],
            out_dim,
            params["heads"],
            params["dropout"],
            params["num_layers"],
            params["use_skip"],
        )
    elif model_name == "GATv2":
        return GATv2Model(
            in_dim,
            params["hidden_dim"],
            out_dim,
            params["heads"],
            params["dropout"],
            params["num_layers"],
        )
    raise ValueError(f"未知模型（gnn_worker.py）: {model_name}")


# ── Worker函数（spawn子进程的入口）────────────────────────────


def worker_process(
    gpu_id, task_queue, result_queue, data_dict, model_name, class_names_list
):
    """
    每个GPU对应一个独立进程。
    从task_queue取 (idx, params)，训练完把结果放入result_queue。
    收到None表示任务结束。
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    # 重建PyG Data
    data = Data(
        x=data_dict["x"],
        edge_index=data_dict["edge_index"],
        y=data_dict["y"],
        train_mask=data_dict["train_mask"],
        val_mask=data_dict["val_mask"],
        test_mask=data_dict["test_mask"],
    )

    in_dim = data.num_features
    out_dim = len(class_names_list)

    print(f"[GPU {gpu_id}] Worker已启动", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[GPU {gpu_id}] 收到终止信号，退出", flush=True)
            break

        idx, params = task
        try:
            print(f"[GPU {gpu_id}] 开始任务 #{idx}: {params}", flush=True)

            model = build_model(model_name, params, in_dim, out_dim)
            num_params = sum(p.numel() for p in model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

            (
                trained_model,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                best_val_acc,
            ) = train_model(
                model, data, device, optimizer, epochs=500, patience=50, verbose=False
            )

            eval_results = evaluate_model(trained_model, data, device)
            config = {
                **params,
                "best_val_acc": best_val_acc,
                "num_params": num_params,
                **eval_results,
            }
            training_history = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }

            save_model_and_results(trained_model, config, model_name, training_history)
            save_detailed_results(
                model_name,
                config,
                training_history,
                eval_results["test_true"],
                eval_results["test_pred"],
                class_names_list,
            )

            result_queue.put((idx, config, None))
            print(
                f"[GPU {gpu_id}] 完成任务 #{idx} | Val={best_val_acc:.4f} Test={eval_results['test_acc']:.4f}",
                flush=True,
            )

        except Exception as e:
            import traceback

            result_queue.put((idx, None, traceback.format_exc()))
            print(f"[GPU {gpu_id}] 任务 #{idx} 失败: {e}", flush=True)
        finally:
            try:
                del model, trained_model, optimizer
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()
