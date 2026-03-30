# ============================================
# utils.R - 通用工具函数
# ============================================

# 加载包（带错误处理）
load_packages <- function(packages, verbose = TRUE) {
    if (verbose) cat("📦 加载R包...\n")
    
    success <- c()
    failed <- c()
    
    for (pkg in packages) {
        result <- tryCatch({
            suppressPackageStartupMessages(library(pkg, character.only = TRUE))
            TRUE
        }, error = function(e) {
            FALSE
        })
        
        if (result) {
            success <- c(success, pkg)
        } else {
            failed <- c(failed, pkg)
        }
    }
    
    if (verbose) {
        cat("  ✅ 成功加载:", paste(success, collapse = ", "), "\n")
        if (length(failed) > 0) {
            cat("  ❌ 加载失败:", paste(failed, collapse = ", "), "\n")
            stop("请安装缺失的包")
        }
    }
    
    return(list(success = success, failed = failed))
}

# 带缓存的加载函数
load_or_process <- function(cache_file, process_func, force_reprocess = FALSE, verbose = TRUE) {
    if (!force_reprocess && file.exists(cache_file)) {
        if (verbose) cat("📦 从缓存加载:", basename(cache_file), "\n")
        return(readRDS(cache_file))
    } else {
        if (verbose) {
            if (force_reprocess) cat("🔄 强制重新处理...\n")
            else cat("🔄 首次运行，正在处理...\n")
        }
        
        obj <- tryCatch({
            process_func()
        }, error = function(e) {
            stop("处理失败: ", e$message)
        })
        
        # 确保缓存目录存在
        cache_dir <- dirname(cache_file)
        if (!dir.exists(cache_dir)) {
            dir.create(cache_dir, recursive = TRUE)
        }
        
        saveRDS(obj, cache_file)
        if (verbose) cat("✅ 已保存到:", basename(cache_file), "\n")
        return(obj)
    }
}

# 数据完整性检查
check_data_integrity <- function(obj, step_name, expected_min_cells = 100) {
    if (is.null(obj)) {
        stop(step_name, " 返回了 NULL 对象")
    }
    
    if (!inherits(obj, "Seurat")) {
        stop(step_name, " 返回的不是 Seurat 对象")
    }
    
    if (ncol(obj) == 0) {
        stop(step_name, " 返回了空对象（0个细胞）")
    }
    
    if (ncol(obj) < expected_min_cells) {
        warning(step_name, " 细胞数 (", ncol(obj), ") 少于预期最小值 (", expected_min_cells, ")")
    }
    
    if (nrow(obj) == 0) {
        stop(step_name, " 返回了空对象（0个基因）")
    }
    
    if (params$verbose) cat("✓", step_name, "通过完整性检查 (", ncol(obj), "细胞, ", nrow(obj), "基因)\n")
    return(TRUE)
}

# 错误处理包装器
with_error_handling <- function(expr, step_name, on_error = "stop") {
    result <- tryCatch({
        expr
    }, error = function(e) {
        error_msg <- paste("❌ 错误在", step_name, ":", e$message)
        cat(error_msg, "\n")
        
        if (on_error == "stop") {
            stop(error_msg)
        } else if (on_error == "warn") {
            warning(error_msg)
            return(NULL)
        } else {
            return(NULL)
        }
    })
    
    return(result)
}

# 进度条显示
show_progress <- function(current, total, step_name = "") {
    if (!params$verbose) return()
    
    percent <- round(current / total * 100, 1)
    cat(sprintf("\r  %s: %.1f%% 完成 (%d/%d)", step_name, percent, current, total))
    if (current == total) cat("\n")
}

# 内存使用监控
monitor_memory <- function(step_name = "") {
    if (!params$verbose) return()
    
    mem_used <- utils::memory.size() / 1024  # MB
    cat(sprintf("  💾 %s - 内存使用: %.2f GB\n", step_name, mem_used / 1024))
}

# 时间记录装饰器
time_it <- function(func, ...) {
    start_time <- Sys.time()
    result <- func(...)
    end_time <- Sys.time()
    elapsed <- difftime(end_time, start_time, units = "auto")
    
    if (params$verbose) {
        cat(sprintf("  ⏱️  耗时: %.2f %s\n", 
                    elapsed, units(elapsed)))
    }
    
    return(result)
}

# 批量处理函数
batch_process <- function(items, process_func, batch_size = 1000, parallel = FALSE) {
    if (parallel) {
        # 并行处理
        library(future)
        plan("multisession")
        
        results <- future.apply::future_lapply(items, function(x) {
            process_func(x)
        })
        
        plan("sequential")
        return(results)
    } else {
        # 串行处理
        results <- list()
        total <- length(items)
        
        for (i in seq_along(items)) {
            results[[i]] <- process_func(items[[i]])
            if (i %% batch_size == 0) {
                show_progress(i, total, "批量处理")
            }
        }
        
        show_progress(total, total, "批量处理")
        return(results)
    }
}

# 确保目录存在
ensure_directory <- function(path) {
    if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
        cat("📁 创建目录:", path, "\n")
    }
    return(path)
}

# 安全读取CSV
safe_read_csv <- function(file, ...) {
    if (!file.exists(file)) {
        stop("文件不存在: ", file)
    }
    
    tryCatch({
        read.csv(file, ...)
    }, error = function(e) {
        stop("无法读取CSV文件: ", e$message)
    })
}

# 格式化数字
format_number <- function(x) {
    format(x, big.mark = ",", scientific = FALSE)
}

# 百分比格式化
format_percent <- function(x, total) {
    sprintf("%.1f%%", x / total * 100)
}