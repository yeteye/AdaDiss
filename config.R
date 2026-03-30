# ============================================
# config.R - 所有配置集中管理
# ============================================

# 1. 定义所有文件路径
paths <- list(
    # ===== 输入数据路径 =====
    # Flex 数据
    flex_h5 = "./data/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5",
    flex_annotation = "./data/FLEX_Ovarian_Barcode_Cluster_Annotation.csv",
    flex_bpcells_dir = "./data/flex_counts_bpcells/",
    flex_cache = "./data/flex_data_processed.rds",
    
    # Xenium 数据
    xenium_dir = "./data/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs/",
    xenium_bpcells_dir = "./data/xenium_counts_bpcells/",
    xenium_cache = "./data/xenium_data_processed.rds",
    xenium_gene_panel = "./data/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs/gene_panel.json",
    
    # ===== 输出数据路径 =====
    output_csv = "./results/cell_groups.csv",
    output_full_csv = "./results/cell_predictions_full.csv",
    output_rds = "./results/xenium_annotated_final.rds",
    output_flex_rds = "./results/flex_reference.rds",
    output_stats = "./results/prediction_stats.txt",
    output_plots = "./results/plots/"
)

# 2. 设置分析参数
params <- list(
    # Flex QC 参数
    flex_min_counts = 200,
    flex_max_counts = 10000,
    flex_max_mt = 10,
    
    # Xenium 参数
    xenium_npcs = 50,
    xenium_dims = 1:30,
    xenium_resolution = 0.6,
    xenium_cluster_name = "clusters",
    
    # 标签转移参数
    transfer_dims = 1:30,
    transfer_k = 50,
    
    # 可视化参数
    seed = 42,
    
    # 输出控制
    save_intermediate = TRUE,
    verbose = TRUE,
    
    # 质量控制阈值
    min_prediction_score = 0.5,
    max_low_score_percentage = 10
)

# 3. 细胞类型映射表
cell_type_mapping <- c(
    "Tumor.Associated.Fibroblasts" = "Tumor Associated Fibroblasts",
    "Endothelial.Cells" = "Endothelial Cells",
    "Stromal.Associated.Fibroblasts" = "Stromal Associated Fibroblasts",
    "T...NK.Cells" = "T & NK Cells",
    "Malignant.Cells.Lining.Cyst" = "Malignant Cells Lining Cyst",
    "Proliferative.Tumor.Cells" = "Proliferative Tumor Cells",
    "Tumor.Cells" = "Tumor Cells",
    "Pericytes" = "Pericytes",
    "Granulosa.Cells" = "Granulosa Cells",
    "Macrophages" = "Macrophages",
    "MT.High..Jun..Fos..Tumor.Cells" = "MT-High, Jun+/Fos+ Tumor Cells",
    "VEGFA..Tumor.Cells" = "VEGFA+ Tumor Cells",
    "Smooth.Muscle.Cells" = "Smooth Muscle Cells",
    "Inflammatory.Tumor.Cells" = "Inflammatory Tumor Cells",
    "Ciliated.Epithelial.Cells" = "Ciliated Epithelial Cells",
    "Fallopian.Tube.Epithelium" = "Fallopian Tube Epithelium"
)

# 反向映射表
reverse_mapping <- setNames(names(cell_type_mapping), cell_type_mapping)

# 4. 验证配置
validate_config <- function() {
    cat("🔍 验证配置...\n")
    errors <- c()
    
    # 检查输入文件存在性
    if (!file.exists(paths$flex_h5)) {
        errors <- c(errors, paste("Flex H5 file not found:", paths$flex_h5))
    }
    if (!file.exists(paths$flex_annotation)) {
        errors <- c(errors, paste("Flex annotation file not found:", paths$flex_annotation))
    }
    if (!dir.exists(paths$xenium_dir)) {
        errors <- c(errors, paste("Xenium directory not found:", paths$xenium_dir))
    }
    
    # 检查参数有效性
    if (params$flex_min_counts >= params$flex_max_counts) {
        errors <- c(errors, "flex_min_counts must be less than flex_max_counts")
    }
    if (params$min_prediction_score < 0 || params$min_prediction_score > 1) {
        errors <- c(errors, "min_prediction_score must be between 0 and 1")
    }
    
    if (length(errors) > 0) {
        cat("❌ 配置验证失败:\n")
        for (err in errors) {
            cat("  - ", err, "\n")
        }
        stop("请修复配置错误后重试")
    }
    
    cat("✅ 配置验证通过\n")
    return(TRUE)
}

# 创建输出目录
create_output_dirs <- function() {
    dirs <- c(dirname(paths$output_csv), paths$output_plots)
    for (dir in dirs) {
        if (!dir.exists(dir)) {
            dir.create(dir, recursive = TRUE)
            cat("📁 创建目录:", dir, "\n")
        }
    }
}

# 打印配置信息
print_config <- function() {
    cat("\n📋 配置信息:\n")
    cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    cat("📂 输入路径:\n")
    cat("  - Flex数据:", basename(paths$flex_h5), "\n")
    cat("  - Flex注释:", basename(paths$flex_annotation), "\n")
    cat("  - Xenium数据:", basename(paths$xenium_dir), "\n")
    cat("\n📤 输出路径:\n")
    cat("  - 预测结果:", basename(paths$output_csv), "\n")
    cat("  - 统计报告:", basename(paths$output_stats), "\n")
    cat("  - 图表目录:", basename(paths$output_plots), "\n")
    cat("\n⚙️ 参数设置:\n")
    cat("  - Flex QC: min_counts=", params$flex_min_counts, 
        ", max_counts=", params$flex_max_counts, 
        ", max_mt=", params$flex_max_mt, "\n", sep = "")
    cat("  - 标签转移: dims=", length(params$transfer_dims), 
        ", k=", params$transfer_k, "\n", sep = "")
    cat("  - 最小预测分数阈值:", params$min_prediction_score, "\n")
    cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
}