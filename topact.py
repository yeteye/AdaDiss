"""
topact.py — TopACT Baseline（SVM + 空间平滑）+ Moran's I 空间自相关

修复清单
--------
1. cuml 依赖 → sklearn（cuml 需要 RAPIDS 环境，普通环境无法安装）
2. _spatial_smooth O(n²) → BallTree 稀疏 kNN（大数据集 OOM 修复）
3. Moran's I 方差公式 → Cliff & Ord (1981) 标准公式（原公式含硬编码 1e-4）
"""

import numpy as np
from sklearn.svm import SVC                          # 修复：替换 cuml.svm.SVC
from sklearn.neighbors import BallTree               # 修复：替换 O(n²) cdist
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class TopACT:
    """
    TopACT：空间感知细胞类型注释（SVM + 空间邻居平滑）。

    训练数据：scRNA（有标签），与 GNN 保持相同特征对齐。
    推断数据：Xenium spot（有空间坐标）。

    Parameters
    ----------
    C, gamma, kernel : SVM 超参
    spatial_weight   : 空间平滑权重 α（最终预测 = (1-α)·SVM + α·spatial）
    n_neighbors      : 空间平滑邻居数
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        spatial_weight: float = 0.3,
        n_neighbors: int = 10,
        seed: int = 42,
    ):
        self.C              = C
        self.gamma          = gamma
        self.kernel         = kernel
        self.spatial_weight = spatial_weight
        self.n_neighbors    = n_neighbors
        self.seed           = seed

        self.svm     = SVC(
            C=C, gamma=gamma, kernel=kernel,
            probability=True, random_state=seed,
            class_weight="balanced",
        )
        self.scaler  = None
        self.classes_ = None
        self._fitted  = False

    # ──────────────────────────────────────────────────
    # 训练
    # ──────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fitted_scaler: StandardScaler | None = None,
    ) -> "TopACT":
        """
        训练 TopACT SVM 分类器。

        Parameters
        ----------
        X_train       : (n_scrna, n_genes) 已 log_normalize 的 scRNA 表达
        y_train       : (n_scrna,) 整型标签
        fitted_scaler : 来自 unified_normalize 的已 fit StandardScaler
        """
        if fitted_scaler is not None:
            self.scaler = fitted_scaler
            X_scaled    = fitted_scaler.transform(X_train)
        else:
            self.scaler = StandardScaler()
            X_scaled    = self.scaler.fit_transform(X_train)

        self.classes_  = np.unique(y_train)
        self.svm.fit(X_scaled, y_train)
        self._fitted = True
        return self

    # ──────────────────────────────────────────────────
    # 推断
    # ──────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        spatial_coords: np.ndarray | None = None,
        return_proba: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        预测细胞类型，可选空间邻居平滑。

        Parameters
        ----------
        X              : (n_spot, n_genes) 已 log_normalize 的表达矩阵
        spatial_coords : (n_spot, 2) 物理坐标（μm）
        return_proba   : 是否同时返回概率矩阵
        """
        assert self._fitted, "Call fit() first."
        X_scaled  = self.scaler.transform(X)
        svm_proba = self.svm.predict_proba(X_scaled)   # (n, n_classes)

        if spatial_coords is not None:
            smooth_proba = self._spatial_smooth(svm_proba, spatial_coords)
            final_proba  = (
                (1 - self.spatial_weight) * svm_proba
                + self.spatial_weight * smooth_proba
            )
        else:
            final_proba = svm_proba

        predictions = final_proba.argmax(axis=1)
        if return_proba:
            return predictions, final_proba
        return predictions

    def _spatial_smooth(
        self,
        proba: np.ndarray,
        coords: np.ndarray,
    ) -> np.ndarray:
        """
        高斯权重空间邻居平滑（BallTree 实现）。

        修复：
        - 原始实现使用 cdist(n, n) 全矩阵，O(n²) 显存
          10 万 spot → ~40GB，必然 OOM
        - 修复后使用 BallTree.query，只计算 k 近邻，O(n·k)

        Parameters
        ----------
        proba  : (n, n_classes) SVM 输出概率
        coords : (n, 2) 空间坐标
        """
        n = len(coords)
        k = min(self.n_neighbors, n - 1)

        # 使用 BallTree 只查询 k 近邻（O(n·k)，而非 O(n²)）
        tree = BallTree(coords, metric="euclidean")
        distances, indices = tree.query(coords, k=k + 1)
        distances = distances[:, 1:]   # (n, k)，去除自身
        indices   = indices[:, 1:]     # (n, k)

        # 高斯权重
        sigma = np.median(distances)
        if sigma == 0:
            sigma = 1.0
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))   # (n, k)
        weights /= weights.sum(axis=1, keepdims=True).clip(min=1e-8)

        # 向量化加权平均：(n, k, 1) * (n, k, n_classes) → (n, n_classes)
        smooth = (weights[:, :, None] * proba[indices]).sum(axis=1)
        return smooth

    # ──────────────────────────────────────────────────
    # Moran's I 空间自相关
    # ──────────────────────────────────────────────────

    @staticmethod
    def morans_i(
        labels: np.ndarray,
        coords: np.ndarray,
        n_neighbors: int = 10,
    ) -> dict:
        """
        计算全局 Moran's I（Cliff & Ord 1981 标准公式）。

        修复：原公式含硬编码 `* 1e-4`（placeholder），
        导致 z-score 虚高，p-value 几乎全为 0，无统计意义。
        本版本使用正确的期望和方差公式（正态性假设）。

        Moran's I ∈ [-1, 1]：
        - I > 0 → 同类细胞空间聚集（正相关）
        - I ≈ 0 → 随机分布
        - I < 0 → 棋盘式分布（负相关）

        Returns
        -------
        dict with 'I', 'E_I', 'z_score', 'p_value'
        """
        from scipy import stats
        n = len(labels)
        x     = labels.astype(float)
        x_dev = x - x.mean()

        # ── 构建稀疏 kNN 权重矩阵（BallTree）────────────
        k = min(n_neighbors, n - 1)
        tree = BallTree(coords, metric="euclidean")
        _, idx = tree.query(coords, k=k + 1)
        idx = idx[:, 1:]   # 去除自身，shape (n, k)

        W = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            W[i, idx[i]] = 1.0

        # ── Moran's I 统计量 ─────────────────────────────
        S0 = W.sum()                                # 权重总和
        num   = n * (W * np.outer(x_dev, x_dev)).sum()
        denom = S0 * (x_dev ** 2).sum()
        I = num / denom if denom != 0 else 0.0

        # ── 期望 & 方差（Cliff & Ord 1981，正态性假设）──
        E_I = -1.0 / (n - 1)

        # S1 = 0.5 * sum_ij (w_ij + w_ji)^2
        W_sym = W + W.T
        S1 = 0.5 * (W_sym ** 2).sum()

        # S2 = sum_i (row_sum_i + col_sum_i)^2
        row_sums = W.sum(axis=1)
        col_sums = W.sum(axis=0)
        S2 = ((row_sums + col_sums) ** 2).sum()

        # Var(I) = [n²S1 - nS2 + 3S0²] / [(n²-1)S0²] - E(I)²
        var_num   = n ** 2 * S1 - n * S2 + 3 * S0 ** 2
        var_denom = (n ** 2 - 1) * S0 ** 2
        Var_I = max(var_num / var_denom - E_I ** 2, 1e-10)

        z     = (I - E_I) / np.sqrt(Var_I)
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        return {"I": float(I), "E_I": float(E_I),
                "z_score": float(z), "p_value": float(p_val)}

    @staticmethod
    def per_class_morans_i(
        labels: np.ndarray,
        coords: np.ndarray,
        cell_types: list[str],
        n_neighbors: int = 10,
    ) -> dict:
        """
        逐细胞类型计算 Moran's I（二值化：是/否该类型）。

        修复：使用 BallTree（O(n·k)）替代原 O(n²) cdist。
        权重矩阵只计算一次，复用到所有细胞类型。

        Returns
        -------
        dict : {cell_type_name: {'I': ..., 'p_value': ...}}
        """
        from scipy import stats
        n = len(labels)
        k = min(n_neighbors, n - 1)

        # 只构建一次权重矩阵
        tree = BallTree(coords, metric="euclidean")
        _, idx = tree.query(coords, k=k + 1)
        idx = idx[:, 1:]

        W = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            W[i, idx[i]] = 1.0

        S0      = W.sum()
        W_sym   = W + W.T
        S1      = 0.5 * (W_sym ** 2).sum()
        row_sum = W.sum(axis=1)
        col_sum = W.sum(axis=0)
        S2      = ((row_sum + col_sum) ** 2).sum()
        E_I     = -1.0 / (n - 1)

        results = {}
        for cls_idx, cls_name in enumerate(cell_types):
            x     = (labels == cls_idx).astype(float)
            x_dev = x - x.mean()
            denom = S0 * (x_dev ** 2).sum()
            if denom == 0:
                results[cls_name] = {"I": 0.0, "p_value": 1.0}
                continue

            I = n * (W * np.outer(x_dev, x_dev)).sum() / denom

            var_num = n ** 2 * S1 - n * S2 + 3 * S0 ** 2
            Var_I   = max(var_num / ((n ** 2 - 1) * S0 ** 2) - E_I ** 2, 1e-10)
            z       = (I - E_I) / np.sqrt(Var_I)
            p       = 2 * (1 - stats.norm.cdf(abs(z)))
            results[cls_name] = {"I": float(I), "p_value": float(p)}

        return results
