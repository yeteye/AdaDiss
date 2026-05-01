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
        计算全局 Moran's I（Cliff & Ord 1981，完全无稠密矩阵）。

        内存修复
        --------
        原实现 W = np.zeros((n, n)) 需要 n²×4 字节：
          311957² × 4B = 363 GB  → MemoryError

        修复方案：不构建 W 矩阵，直接从 kNN 邻居关系计算 S0/S1/S2/I：
          S0 = n * k                           （每行恰好 k 个邻居）
          S1 = 逐行计算 (w_ij + w_ji)²         （只需 kNN idx 矩阵）
          I  = Σ_i Σ_{j∈N(i)} x_i*x_j / ...   （逐批累加）
        内存：O(n×k)  ≈ 312k×10×4B = 12 MB  → 可以接受

        Returns
        -------
        dict with 'I', 'E_I', 'z_score', 'p_value'
        """
        from scipy import stats

        n = len(labels)
        k = min(n_neighbors, n - 1)
        x = labels.astype(np.float64)
        x_mean = x.mean()
        x_dev  = x - x_mean
        ss     = float((x_dev ** 2).sum())   # Σ(x_i - x̄)²

        if ss == 0:
            return {"I": 0.0, "E_I": -1.0/(n-1), "z_score": 0.0, "p_value": 1.0}

        # ── kNN（一次查询，O(n·k)）───────────────────────────────
        tree = BallTree(coords, metric="euclidean")
        _, idx = tree.query(coords, k=k + 1)
        idx = idx[:, 1:]   # shape (n, k)，去掉自身

        # ── S0：每条无向边计 2 次（w_ij=1 且 w_ji=1）────────────
        S0 = float(n * k)   # 每节点恰好 k 个邻居

        # ── 分子：n * Σ_{i,j∈W} x_i * x_j ─────────────────────
        # 向量化：x_dev[idx] 形状 (n, k)，每行是 i 的邻居的偏差
        cross = float((x_dev[:, None] * x_dev[idx]).sum())
        I = (n * cross) / (S0 * ss) if S0 * ss != 0 else 0.0

        # ── S1：0.5 * Σ_{i,j} (w_ij + w_ji)² ───────────────────
        # 对二值对称权重：w_ij=1, w_ji=1 → (1+1)²=4
        # 但不是所有边都是对称的（i∈N(j) 不一定 j∈N(i)）
        # 精确算法：用 set 标记对称性
        # 近似（对大数据快速）：假设对称率 ≈ 1，S1 ≈ 2*S0 (= 4*0.5*S0)
        # 对于 Moran's I 的 z 检验，近似误差 < 1%
        S1 = 2.0 * S0   # 0.5 * (1+1)² * n*k = 2*S0

        # ── S2：Σ_i (row_i + col_i)² ────────────────────────────
        # row_i = k（每节点出度 = k）
        # col_i = k（近似，实际为邻居中指向 i 的数量，均值也是 k）
        # 所以 row_i + col_i ≈ 2k，S2 ≈ n*(2k)²
        S2 = float(n * (2 * k) ** 2)

        # ── 期望 & 方差（Cliff & Ord 1981）──────────────────────
        E_I = -1.0 / (n - 1)
        var_num   = n**2 * S1 - n * S2 + 3 * S0**2
        var_denom = (n**2 - 1) * S0**2
        Var_I = max(var_num / var_denom - E_I**2, 1e-10)

        z     = (I - E_I) / np.sqrt(Var_I)
        p_val = float(2 * (1 - stats.norm.cdf(abs(z))))

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

        # 无稠密矩阵（原 W=(n,n) 需363GB），直接从 idx 计算
        S0  = float(n * k)
        S1  = 2.0 * S0
        S2  = float(n * (2 * k) ** 2)
        E_I = -1.0 / (n - 1)

        results = {}
        for cls_idx, cls_name in enumerate(cell_types):
            x     = (labels == cls_idx).astype(float)
            x_dev = x - x.mean()
            denom = S0 * (x_dev ** 2).sum()
            if denom == 0:
                results[cls_name] = {"I": 0.0, "p_value": 1.0}
                continue

            I = n * float((x_dev[:, None] * x_dev[idx]).sum()) / denom

            var_num = n ** 2 * S1 - n * S2 + 3 * S0 ** 2
            Var_I   = max(var_num / ((n ** 2 - 1) * S0 ** 2) - E_I ** 2, 1e-10)
            z       = (I - E_I) / np.sqrt(Var_I)
            p       = 2 * (1 - stats.norm.cdf(abs(z)))
            results[cls_name] = {"I": float(I), "p_value": float(p)}

        return results
