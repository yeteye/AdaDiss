import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1. 生物预处理 + 统一归一化
# =========================
def preprocess_expr(expr):
    expr = expr / (expr.sum(axis=1, keepdims=True) + 1e-8) * 1e4
    expr = np.log1p(expr)
    return expr


flex_expr = preprocess_expr(flex_expr)
xenium_expr = preprocess_expr(xenium_expr)

scaler = StandardScaler()
flex_expr = scaler.fit_transform(flex_expr)
xenium_expr = scaler.transform(xenium_expr)


# =========================
# 2. 构建 mutual kNN 图
# =========================
def build_mutual_knn(features, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    _, indices = nbrs.kneighbors(features)

    edge_dict = {}

    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            key = tuple(sorted((i, j)))
            edge_dict[key] = edge_dict.get(key, 0) + 1

    edges = [list(k) for k, v in edge_dict.items() if v > 1]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return edge_index


# 合并数据
all_expr = np.vstack([flex_expr, xenium_expr])
edge_index = build_mutual_knn(all_expr, k=10)

# =========================
# 3. 构建标签 & mask
# =========================
n_flex = flex_expr.shape[0]
n_xenium = xenium_expr.shape[0]

y = np.concatenate([flex_labels, np.full(n_xenium, -1)])

# 分层采样
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(flex_expr, flex_labels))

train_mask = torch.zeros(len(y), dtype=torch.bool)
val_mask = torch.zeros(len(y), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True

# flex / xenium mask
flex_mask = torch.zeros(len(y), dtype=torch.bool)
xenium_mask = torch.zeros(len(y), dtype=torch.bool)

flex_mask[:n_flex] = True
xenium_mask[n_flex:] = True

# =========================
# 4. PyG Data
# =========================
data = Data(
    x=torch.tensor(all_expr, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long),
)

data.train_mask = train_mask
data.val_mask = val_mask
data.flex_mask = flex_mask
data.xenium_mask = xenium_mask

data = data.to(device)

# =========================
# 5. 类别权重
# =========================
weights = compute_class_weight(
    "balanced", classes=np.unique(flex_labels), y=flex_labels
)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)


# =========================
# 6. 模型（改进版 GCN）
# =========================
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()

        self.lin0 = nn.Linear(in_dim, 512)
        self.conv1 = GCNConv(512, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin0(x)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.lin_out(x)

        return F.log_softmax(x, dim=1)


model = GCN(
    in_dim=data.x.shape[1],
    hidden_dim=256,
    num_classes=len(np.unique(flex_labels)),
    dropout=0.5,
).to(device)


# =========================
# 7. MMD（Domain Adaptation）
# =========================
def gaussian_kernel(x, y, sigma=1.0):
    dist = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
    return torch.exp(-dist / (2 * sigma**2))


def mmd_loss(x, y):
    xx = gaussian_kernel(x, x).mean()
    yy = gaussian_kernel(y, y).mean()
    xy = gaussian_kernel(x, y).mean()
    return xx + yy - 2 * xy


# =========================
# 8. 训练
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
)

best_val_acc = 0
best_model_state = None

lambda_mmd = 0.1
lambda_ent = 0.01

for epoch in range(500):

    model.train()
    optimizer.zero_grad()

    out = model(data)

    # ===== CE loss =====
    ce_loss = F.nll_loss(
        out[data.train_mask], data.y[data.train_mask], weight=class_weights
    )

    # ===== MMD =====
    h = out.exp()

    h_flex = h[data.flex_mask]
    h_xenium = h[data.xenium_mask]

    loss_mmd = mmd_loss(h_flex, h_xenium)

    # ===== Entropy =====
    xenium_logits = out[data.xenium_mask]
    entropy = -(xenium_logits.exp() * xenium_logits).sum(dim=1).mean()

    loss = ce_loss + lambda_mmd * loss_mmd + lambda_ent * entropy

    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # ===== 验证 =====
    model.eval()
    pred = out.argmax(dim=1)

    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())

    scheduler.step(acc)

    if acc > best_val_acc:
        best_val_acc = acc
        best_model_state = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val Acc {acc:.4f}")

# =========================
# 9. 加载最优模型
# =========================
model.load_state_dict(best_model_state)

# =========================
# 10. Xenium 预测
# =========================
model.eval()
out = model(data)

xenium_pred = out[data.xenium_mask].argmax(dim=1).cpu().numpy()

print("Xenium prediction done.")
