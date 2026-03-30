# ============================================
# export.R - 导出功能
# ============================================

# 准备导出数据
prepare_export_data <- function(xenium.obj, paths, params, mapping) {
    cat("📊 准备导出数据...\n")
    
    # 基础数据框
    export_df <- xenium.obj@meta.data %>%
        rownames_to_column(var = "cell_id")
    
    # 添加聚类信息
    if (params$xenium_cluster_name %in% colnames(export_df)) {
        export_df$cluster <- export_df[[params$xenium_cluster_name]]
    }
    
    # 添加预测类型
    export_df$predicted_cell_type <- export_df$predicted.id
    
    # 添加分数列
    score_cols <- grep("prediction\\.score\\.", colnames(export_df), value = TRUE)
    
    if (length(score_cols) > 0) {
        cat("  添加预测分数和一致性检查...\n")
        
        # 计算预测分数
        export_df$prediction_score <- sapply(1:nrow(export_df), function(i) {
            cell_type <- export_df$predicted_cell_type[i]
            if (!is.na(cell_type) && cell_type %in% names(mapping)) {
                score_col <- paste0("prediction.score.", mapping[cell_type])
                if (score_col %in% colnames(export_df)) {
                    return(as.numeric(export_df[i, score_col]))
                }
            }
            return(NA_real_)
        })
        
        # 计算最高分类型
        export_df$max_score_type <- sapply(1:nrow(export_df), function(i) {
            scores <- as.numeric(export_df[i, score_cols])
            if (all(is.na(scores))) return(NA)
            max_idx <- which.max(scores)
            if (length(max_idx) > 0) {
                max_type_key <- gsub("prediction.score.", "", score_cols[max_idx])
                if (max_type_key %in% names(mapping)) {
                    return(mapping[max_type_key])
                }
            }
            return(NA)
        })
        
        # 一致性检查
        export_df$prediction_consistent <- export_df$predicted_cell_type == export_df$max_score_type
        
        cat("  ✅ 预测一致性: ", 
            format_percent(sum(export_df$prediction_consistent, na.rm = TRUE), 
                          sum(!is.na(export_df$prediction_consistent))), "\n", sep = "")
    }
    
    return(export_df)
}

# 导出基础CSV
export_basic_csv <- function(export_df, output_path) {
    cat("📄 导出基础CSV...\n")
    
    # 选择基础列
    base_cols <- c("cell_id")
    
    if ("cluster" %in% colnames(export_df)) {
        base_cols <- c(base_cols, "cluster")
    }
    
    base_cols <- c(base_cols, "predicted_cell_type", "prediction_score")
    
    if ("max_score_type" %in% colnames(export_df)) {
        base_cols <- c(base_cols, "max_score_type", "prediction_consistent")
    }
    
    basic_df <- export_df %>% select(all_of(base_cols))
    
    write.csv(basic_df, output_path, row.names = FALSE)
    cat("  ✅ 已保存:", output_path, "\n")
    cat("     - 细胞数:", format_number(nrow(basic_df)), "\n")
    cat("     - 列数:", ncol(basic_df), "\n")
}

# 导出完整CSV（含所有分数）
export_full_csv <- function(export_df, output_path) {
    cat("📄 导出完整CSV...\n")
    
    score_cols <- grep("prediction\\.score\\.", colnames(export_df), value = TRUE)
    
    if (length(score_cols) > 0) {
        full_df <- export_df %>%
            select(cell_id, predicted_cell_type, prediction_score,
                   max_score_type, prediction_consistent, all_of(score_cols))
        
        write.csv(full_df, output_path, row.names = FALSE)
        cat("  ✅ 已保存:", output_path, "\n")
        cat("     - 细胞数:", format_number(nrow(full_df)), "\n")
        cat("     - 列数:", ncol(full_df), "\n")
    } else {
        cat("  ⚠️ 未找到分数列，跳过完整CSV导出\n")
    }
}

# 生成统计报告
generate_stat_report <- function(export_df, output_path, params) {
    cat("📊 生成统计报告...\n")
    
    sink(output_path)
    
    cat("=", rep("=", 58), "\n", sep = "")
    cat("细胞类型预测统计报告\n")
    cat("=", rep("=", 58), "\n", sep = "")
    cat("生成时间:", Sys.time(), "\n")
    cat("总细胞数:", format_number(nrow(export_df)), "\n")
    cat("=", rep("=", 58), "\n\n", sep = "")
    
    # 1. 细胞类型分布
    cat("📈 1. 预测细胞类型分布\n")
    cat("-", rep("-", 40), "\n", sep = "")
    type_counts <- sort(table(export_df$predicted_cell_type), decreasing = TRUE)
    for (type in names(type_counts)) {
        count <- type_counts[type]
        percent <- count / nrow(export_df) * 100
        cat(sprintf("  %-35s: %8s (%5.1f%%)\n", type, format_number(count), percent))
    }
    
    # 2. 预测分数统计
    if ("prediction_score" %in% colnames(export_df)) {
        valid_scores <- export_df$prediction_score[!is.na(export_df$prediction_score)]
        
        cat("\n📈 2. 预测分数统计\n")
        cat("-", rep("-", 40), "\n", sep = "")
        cat("  有效分数细胞数:", format_number(length(valid_scores)), "\n")
        cat("  分数范围:", round(min(valid_scores), 4), "-", round(max(valid_scores), 4), "\n")
        cat("  均值:", round(mean(valid_scores), 4), "\n")
        cat("  中位数:", round(median(valid_scores), 4), "\n")
        cat("  标准差:", round(sd(valid_scores), 4), "\n")
        
        # 分数区间分布
        cat("\n  分数区间分布:\n")
        breaks <- seq(0, 1, by = 0.1)
        score_bins <- cut(valid_scores, breaks, include.lowest = TRUE)
        bin_counts <- table(score_bins)
        for (i in seq_along(bin_counts)) {
            cat(sprintf("    %s: %8s (%5.1f%%)\n", 
                       names(bin_counts)[i],
                       format_number(bin_counts[i]),
                       bin_counts[i] / length(valid_scores) * 100))
        }
    }
    
    # 3. 按细胞类型的分数统计
    if ("prediction_score" %in% colnames(export_df)) {
        cat("\n📈 3. 按细胞类型的预测分数统计\n")
        cat("-", rep("-", 40), "\n", sep = "")
        
        score_by_type <- export_df %>%
            filter(!is.na(prediction_score)) %>%
            group_by(predicted_cell_type) %>%
            summarise(
                n_cells = n(),
                mean_score = mean(prediction_score, na.rm = TRUE),
                median_score = median(prediction_score, na.rm = TRUE),
                sd_score = sd(prediction_score, na.rm = TRUE),
                min_score = min(prediction_score, na.rm = TRUE),
                max_score = max(prediction_score, na.rm = TRUE),
                .groups = 'drop'
            ) %>%
            arrange(desc(mean_score))
        
        for (i in 1:nrow(score_by_type)) {
            row <- score_by_type[i, ]
            cat(sprintf("  %-35s: n=%6s, 均值=%.3f, 中位数=%.3f\n",
                       row$predicted_cell_type,
                       format_number(row$n_cells),
                       row$mean_score,
                       row$median_score))
        }
    }
    
    # 4. 一致性统计
    if ("prediction_consistent" %in% colnames(export_df)) {
        cat("\n📈 4. 预测一致性统计\n")
        cat("-", rep("-", 40), "\n", sep = "")
        
        consistent_count <- sum(export_df$prediction_consistent, na.rm = TRUE)
        total_valid <- sum(!is.na(export_df$prediction_consistent))
        consistent_rate <- consistent_count / total_valid * 100
        
        cat("  一致细胞数:", format_number(consistent_count), "\n")
        cat("  不一致细胞数:", format_number(total_valid - consistent_count), "\n")
        cat("  一致率:", sprintf("%.2f%%", consistent_rate), "\n")
        
        # 按类型的一致性
        cat("\n  按细胞类型的一致性:\n")
        consistency_by_type <- export_df %>%
            filter(!is.na(prediction_consistent)) %>%
            group_by(predicted_cell_type) %>%
            summarise(
                n_cells = n(),
                consistent = sum(prediction_consistent, na.rm = TRUE),
                consistency_rate = consistent / n() * 100,
                .groups = 'drop'
            ) %>%
            arrange(desc(consistency_rate))
        
        for (i in 1:nrow(consistency_by_type)) {
            row <- consistency_by_type[i, ]
            cat(sprintf("    %-35s: %6s/%6s (%5.1f%%)\n",
                       row$predicted_cell_type,
                       format_number(row$consistent),
                       format_number(row$n_cells),
                       row$consistency_rate))
        }
        
        # 不一致案例
        inconsistent_cases <- export_df %>%
            filter(!prediction_consistent & !is.na(prediction_consistent)) %>%
            select(cell_id, predicted_cell_type, max_score_type, prediction_score) %>%
            head(20)
        
        if (nrow(inconsistent_cases) > 0) {
            cat("\n  不一致案例示例（前20个）:\n")
            print(inconsistent_cases)
        }
    }
    
    # 5. 质量控制
    cat("\n📈 5. 质量控制\n")
    cat("-", rep("-", 40), "\n", sep = "")
    
    if ("prediction_score" %in% colnames(export_df)) {
        low_score_cells <- sum(export_df$prediction_score < params$min_prediction_score, na.rm = TRUE)
        low_score_percent <- low_score_cells / nrow(export_df) * 100
        
        cat("  低分细胞数 (<", params$min_prediction_score, "): ", 
            format_number(low_score_cells), " (", sprintf("%.1f%%", low_score_percent), ")\n", sep = "")
        
        if (low_score_percent > params$max_low_score_percentage) {
            cat("  ⚠️ 警告: 低分细胞比例超过阈值", params$max_low_score_percentage, "%\n")
        } else {
            cat("  ✅ 低分细胞比例在可接受范围内\n")
        }
    }
    
    sink()
    
    cat("  ✅ 统计报告已保存:", output_path, "\n")
}

# 导出Seurat对象
export_seurat_object <- function(obj, output_path, compress = TRUE) {
    cat("💾 保存Seurat对象...\n")
    
    tryCatch({
        if (compress) {
            saveRDS(obj, output_path, compress = "gzip")
        } else {
            saveRDS(obj, output_path)
        }
        cat("  ✅ 已保存:", output_path, "\n")
        cat("     - 对象大小:", format(file.info(output_path)$size / 1e6, digits = 2), "MB\n")
    }, error = function(e) {
        cat("  ❌ 保存失败:", e$message, "\n")
    })
}

# 主导出函数
export_results <- function(xenium.obj, flex_data.obj, paths, params, mapping) {
    cat("\n" + paste(rep("━", 60), collapse = ""), "\n")
    cat("📤 开始导出结果\n")
    cat(paste(rep("━", 60), collapse = ""), "\n")
    
    # 1. 准备数据
    export_df <- prepare_export_data(xenium.obj, paths, params, mapping)
    
    # 2. 导出CSV文件
    export_basic_csv(export_df, paths$output_csv)
    export_full_csv(export_df, paths$output_full_csv)
    
    # 3. 导出Seurat对象
    export_seurat_object(xenium.obj, paths$output_rds)
    
    # 4. 导出Flex参考对象
    if (params$save_intermediate && !is.null(flex_data.obj)) {
        export_seurat_object(flex_data.obj, paths$output_flex_rds)
    }
    
    # 5. 生成统计报告
    generate_stat_report(export_df, paths$output_stats, params)
    
    # 6. 显示摘要
    cat("\n", paste(rep("━", 60), collapse = ""), "\n")
    cat("📊 结果摘要\n")
    cat(paste(rep("━", 60), collapse = ""), "\n")
    cat("  总细胞数:", format_number(nrow(export_df)), "\n")
    
    if ("prediction_score" %in% colnames(export_df)) {
        valid_scores <- export_df$prediction_score[!is.na(export_df$prediction_score)]
        cat("  有效分数细胞数:", format_number(length(valid_scores)), "\n")
        cat("  中位数分数:", round(median(valid_scores), 3), "\n")
        
        low_score_count <- sum(export_df$prediction_score < params$min_prediction_score, na.rm = TRUE)
        cat("  低分细胞 (<", params$min_prediction_score, "): ", 
            format_number(low_score_count), " (", 
            sprintf("%.1f%%", low_score_count/nrow(export_df)*100), ")\n", sep = "")
    }
    
    if ("prediction_consistent" %in% colnames(export_df)) {
        consistent_rate <- sum(export_df$prediction_consistent, na.rm = TRUE) / 
                          sum(!is.na(export_df$prediction_consistent)) * 100
        cat("  预测一致性: ", sprintf("%.1f%%", consistent_rate), "\n", sep = "")
    }
    
    cat(paste(rep("━", 60), collapse = ""), "\n")
    cat("✅ 导出完成\n\n")
    
    return(export_df)
}