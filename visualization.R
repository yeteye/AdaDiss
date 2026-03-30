# ============================================
# visualization.R - 可视化函数
# ============================================

# 创建统一的UMAP图
create_dim_plot <- function(obj, reduction = "umap", group_by, 
                            title = "", pt_size = 0.3, label = TRUE,
                            colors = NULL) {
    p <- DimPlot(obj, reduction = reduction, group.by = group_by,
                 pt.size = pt_size, label = label, repel = TRUE)
    
    if (!is.null(colors)) {
        p <- p + scale_color_manual(values = colors)
    }
    
    if (title != "") {
        p <- p + ggtitle(title)
    }
    
    return(p)
}

# 创建空间图
create_spatial_plot <- function(obj, fov = "fov", group_by, 
                                title = "", size = 0.5, 
                                dark_background = FALSE) {
    p <- tryCatch({
        ImageDimPlot(obj, fov = fov, group.by = group_by,
                     size = size, dark.background = dark_background) +
            ggtitle(title)
    }, error = function(e) {
        cat("  空间可视化失败:", e$message, "\n")
        return(NULL)
    })
    
    return(p)
}

# 创建预测分数小提琴图
create_score_violin <- function(meta_data, score_col = "prediction_score", 
                                group_col = "predicted.id", 
                                title = "Prediction Scores by Cell Type") {
    
    # 过滤NA值
    plot_data <- meta_data %>%
        filter(!is.na(.data[[score_col]]), !is.na(.data[[group_col]]))
    
    p <- ggplot(plot_data, aes(x = .data[[group_col]], 
                                y = .data[[score_col]], 
                                fill = .data[[group_col]])) +
        geom_violin(scale = "width", trim = TRUE) +
        stat_summary(fun = median, geom = "point", size = 0.5, color = "black") +
        scale_fill_viridis_d(alpha = 0.7) +
        theme_minimal() +
        theme(legend.position = "none",
              axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
              plot.title = element_text(hjust = 0.5)) +
        ggtitle(title) +
        xlab("") + ylab("Prediction Score") +
        ylim(0, 1)
    
    return(p)
}

# 创建分数分布直方图
create_score_histogram <- function(meta_data, score_col = "prediction_score",
                                   bins = 50, title = "Prediction Score Distribution") {
    
    plot_data <- meta_data %>% filter(!is.na(.data[[score_col]]))
    
    p <- ggplot(plot_data, aes(x = .data[[score_col]])) +
        geom_histogram(bins = bins, fill = "steelblue", alpha = 0.7) +
        theme_minimal() +
        ggtitle(title) +
        xlab("Prediction Score") + ylab("Number of Cells") +
        xlim(0, 1)
    
    return(p)
}

# 创建相关性散点图
create_correlation_plot <- function(data, x_col, y_col, color_col = NULL,
                                     title = "Gene Expression Correlation") {
    
    p <- ggplot(data, aes(x = .data[[x_col]], y = .data[[y_col]]))
    
    if (!is.null(color_col)) {
        p <- p + geom_point(aes(color = .data[[color_col]]), size = 0.5) +
            scale_colour_manual(values = c("darkcyan", "coral"))
    } else {
        p <- p + geom_point(size = 0.5, alpha = 0.6)
    }
    
    p <- p +
        stat_poly_eq() +
        scale_x_log10() + scale_y_log10() +
        xlab("SC Flex Mean Expression") + ylab("Xenium Mean Expression") +
        ggtitle(title) +
        theme_classic() +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.5)
    
    return(p)
}

# 创建组合图
create_composite_plot <- function(plots, ncol = 2, title = NULL) {
    combined <- wrap_plots(plots, ncol = ncol)
    
    if (!is.null(title)) {
        combined <- combined + plot_annotation(title = title)
    }
    
    return(combined)
}

# 保存所有图表
save_plots <- function(plots, output_dir, prefix = "plot", 
                       width = 10, height = 8, dpi = 300) {
    
    ensure_directory(output_dir)
    
    for (name in names(plots)) {
        if (!is.null(plots[[name]])) {
            filename <- file.path(output_dir, paste0(prefix, "_", name, ".png"))
            ggsave(filename, plots[[name]], width = width, height = height, dpi = dpi)
            cat("  💾 保存图表:", basename(filename), "\n")
        }
    }
}

# 批量生成可视化
generate_visualizations <- function(xenium.obj, flex_data.obj, 
                                    output_dir, params) {
    
    plots <- list()
    
    # Flex可视化
    if (!is.null(flex_data.obj)) {
        plots$flex_clusters <- create_dim_plot(
            flex_data.obj, group_by = "RNA_snn_res.0.5",
            title = "Flex Clusters"
        )
        
        plots$flex_cell_types <- create_dim_plot(
            flex_data.obj, group_by = "cell_type",
            title = "Flex Cell Types"
        )
    }
    
    # Xenium可视化
    if ("predicted.id" %in% colnames(xenium.obj@meta.data)) {
        plots$xenium_predicted_umap <- create_dim_plot(
            xenium.obj, group_by = "predicted.id",
            title = "Predicted Cell Types - UMAP"
        )
        
        plots$xenium_predicted_spatial <- create_spatial_plot(
            xenium.obj, group_by = "predicted.id",
            title = "Predicted Cell Types - Spatial"
        )
        
        if ("prediction_score" %in% colnames(xenium.obj@meta.data)) {
            plots$score_violin <- create_score_violin(
                xenium.obj@meta.data,
                title = "Prediction Scores by Cell Type"
            )
            
            plots$score_histogram <- create_score_histogram(
                xenium.obj@meta.data,
                title = "Overall Prediction Score Distribution"
            )
        }
    }
    
    # Xenium原始聚类可视化
    if (params$xenium_cluster_name %in% colnames(xenium.obj@meta.data)) {
        plots$xenium_clusters_umap <- create_dim_plot(
            xenium.obj, group_by = params$xenium_cluster_name,
            title = "Xenium Clusters"
        )
        
        plots$xenium_clusters_spatial <- create_spatial_plot(
            xenium.obj, group_by = params$xenium_cluster_name,
            title = "Xenium Spatial Clusters"
        )
    }
    
    return(plots)
}