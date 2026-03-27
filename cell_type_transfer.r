# 安装必要包（只需运行一次）
install.packages('remotes')
remotes::install_github("mojaveazure/seurat-disk")
remotes::install_github("bnprks/BPCells/r")
install.packages(c("Seurat", "SeuratObject", "arrow", "tidyverse", "jsonlite", 
                   "ggplot2", "ggpmisc", "scales", "cowplot", "gridExtra", 
                   "viridis", "hrbrthemes"))

# 加载包
library(Seurat)
library(BPCells)
library(SeuratObject)
library(SeuratDisk)
library(tidyverse)
library(jsonlite)
options(future.globals.maxSize = 1e9)

# 绘图包
library(ggplot2)
library(ggpmisc)
library(scales)
library(cowplot)
library(gridExtra)
library(viridis)
library(hrbrthemes)