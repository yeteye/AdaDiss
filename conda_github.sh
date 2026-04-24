conda run -n spatial_gnn --no-capture-output Rscript - <<'RSCRIPT'
# 使用清华 CRAN 镜像，不访问 GitHub
options(repos = c(
    CRAN     = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/",
    BioC     = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor/"
))

# IRkernel（从 CRAN 安装，不需要 GitHub）
if (!requireNamespace("IRkernel", quietly = TRUE)) {
    install.packages("IRkernel")
}
IRkernel::installspec(name = "spatial_gnn_R", displayname = "R (spatial_gnn)")
cat("IRkernel registered OK\n")

# SeuratDisk：CRAN 上有官方版本，不需要 GitHub
if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
    install.packages("SeuratDisk")
}
cat("SeuratDisk OK\n")

# BPCells：只有 GitHub 版，跳过（你的数据量不需要 on-disk 矩阵）
cat("BPCells skipped (GitHub unreachable, not required for this project)\n")

# 验证已安装的核心包
for (pkg in c("Seurat", "SeuratObject", "IRkernel", "dplyr", "ggplot2", "Matrix")) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("  ✓ %-15s %s\n", pkg, packageVersion(pkg)))
    } else {
        cat(sprintf("  ✗ %-15s NOT FOUND\n", pkg))
    }
}
RSCRIPT