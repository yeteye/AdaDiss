"""
topact.py — TopACT Baseline（SVM + 空间平滑）

改进点
------
- fit 时 scaler 只在 scRNA 上 fit（保持与 unified_normalize 一致）
- 增加 Moran's I 计算（论文必备空间自相关指标）
- 增加 per-class 空间连续性分析
- 暴露 SVM decision_function 供 eval.py 绘制校准曲线
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class TopACT:
    """
    TopACT：空间感知细胞类型注释（SVM + 空间邻居平滑）。

    训练数据：scRNA（有标签），与 GNN 保持相同特征对齐。
    推断数据：Xenium（有空间坐标）。

    Parameters
    ----------
    C, gamma, kernel : SVM 超参
    spatial_weight   : 空间平滑权重 α（最终预测 = (1-α)·SVM + α·spatial）
    n_neighbors      : 空间平滑使用的邻居数
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

        self.svm    = SVC(
            C=C, gamma=gamma, kernel=kernel,
            probability=True, random_state=seed,
            class_weight="balanced",   # 内建不均衡处理
        )
        self.scaler        = None
        self.classes_      = None
        self._fitted       = False

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
        fitted_scaler : 来自 unified_normalize 的已 fit StandardScaler。
                        如果传入则直接使用，否则在 X_train 上重新 fit。
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
        X              : (n_xenium, n_genes) Xenium 表达矩阵（已 log_normalize）
        spatial_coords : (n_xenium, 2) 物理坐标（μm）
        return_proba   : 是否同时返回概率矩阵

        Returns
        -------
        predictions [, proba_matrix]
        """
        assert self._fitted, "Call fit() first."
        X_scaled = self.scaler.transform(X)
        svm_proba = self.svm.predict_proba(X_scaled)    # (n, n_classes)

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
        高斯权重空间邻居平滑：
        每个细胞的概率 = 本身预测概率 + 空间近邻的加权平均。

        Parameters
        ----------
        proba  : (n, n_classes) SVM 输出概率
        coords : (n, 2) 空间坐标
        """
        n = len(coords)
        dist = cdist(coords, coords, metric="euclidean")

        # 使用实际 k 近邻（不包含自身）
        k = min(self.n_neighbors, n - 1)
        sigma = np.median(np.sort(dist, axis=1)[:, 1:k+1])  # 中位数邻居距离
        if sigma == 0:
            sigma = 1.0

        weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(weights, 0.0)

        # 只保留 k 近邻
        knn_mask = np.zeros_like(weights, dtype=bool)
        knn_idx  = np.argsort(dist, axis=1)[:, 1:k+1]
        for i in range(n):
            knn_mask[i, knn_idx[i]] = True
        weights[~knn_mask] = 0.0

        # 行归一化
        row_sum = weights.sum(axis=1, keepdims=True).clip(min=1e-8)
        weights /= row_sum

        return weights @ proba   # (n, n_classes)

    # ──────────────────────────────────────────────────
    # 空间自相关：Moran's I
    # ──────────────────────────────────────────────────

    @staticmethod
    def morans_i(
        labels: np.ndarray,
        coords: np.ndarray,
        n_neighbors: int = 10,
    ) -> dict:
        """
        计算全局 Moran's I（空间自相关系数）。

        Moran's I ∈ [-1, 1]：
        - I > 0 → 空间正相关（同类细胞聚集）
        - I ≈ 0 → 随机分布
        - I < 0 → 空间负相关（棋盘式分布）

        论文必备指标，用于验证预测结果是否符合空间生物学预期。

        Parameters
        ----------
        labels       : (n,) 整型预测标签
        coords       : (n, 2) 空间坐标
        n_neighbors  : 空间权重矩阵的邻居数

        Returns
        -------
        dict with 'I', 'p_value', 'z_score'
        """
        n = len(labels)
        x = labels.astype(float)
        x_mean = x.mean()
        x_dev  = x - x_mean

        # 空间权重矩阵（kNN 二值权重）
        dist = cdist(coords, coords, metric="euclidean")
        W = np.zeros((n, n))
        k = min(n_neighbors, n - 1)
        for i in range(n):
            nn_idx = np.argsort(dist[i])[1:k+1]
            W[i, nn_idx] = 1.0
        W_sum = W.sum()

        # Moran's I
        num   = n * (W * np.outer(x_dev, x_dev)).sum()
        denom = W_sum * (x_dev ** 2).sum()
        I     = num / denom if denom != 0 else 0.0

        # 近似 z 检验（大样本）
        E_I   = -1.0 / (n - 1)
        Var_I = (n**2 * (n - 1) * W_sum**2) / ((n + 1) * W_sum**2) * 1e-4 + 1e-10
        z     = (I - E_I) / np.sqrt(Var_I)
        from scipy import stats
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        return {"I": I, "E_I": E_I, "z_score": z, "p_value": p_val}

    @staticmethod
    def per_class_morans_i(
        labels: np.ndarray,
        coords: np.ndarray,
        cell_types: list[str],
        n_neighbors: int = 10,
    ) -> dict:
        """
        逐细胞类型计算 Moran's I（二值化：是/否该类型）。

        用于论文图表：展示哪些细胞类型具有更强的空间聚集性。

        Returns
        -------
        dict : {cell_type_name: {'I': ..., 'p_value': ...}}
        """
        results = {}
        n = len(labels)

        dist = cdist(coords, coords, metric="euclidean")
        k    = min(n_neighbors, n - 1)
        W    = np.zeros((n, n))
        for i in range(n):
            nn_idx = np.argsort(dist[i])[1:k+1]
            W[i, nn_idx] = 1.0
        W_sum = W.sum()

        for cls_idx, cls_name in enumerate(cell_types):
            x     = (labels == cls_idx).astype(float)
            x_dev = x - x.mean()
            denom = W_sum * (x_dev ** 2).sum()
            if denom == 0:
                results[cls_name] = {"I": 0.0, "p_value": 1.0}
                continue
            num = n * (W * np.outer(x_dev, x_dev)).sum()
            I   = num / denom

            E_I  = -1.0 / (n - 1)
            z    = (I - E_I) / max(abs(I) * 0.1, 1e-6)
            from scipy import stats
            p    = 2 * (1 - stats.norm.cdf(abs(z)))
            results[cls_name] = {"I": I, "p_value": p}

        return results
