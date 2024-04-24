
library(kBET)
library(anndata)

# anndata to evaluate
in_anndata <- '/home/users/chhy/jnb/scvi_sub.h5ad'
ad <- read_h5ad(in_anndata)
batch <- ad$obs$batch
clusters <- ad$obs$ident
data <- ad$X

#data: a matrix (rows: samples, columns: features (genes))
#batch: vector or factor with batch label of each cell 
#clusters: vector or factor with cluster label of each cell 

kBET_result_list <- list()
sum_kBET <- 0
dic_kBET_result <- {}

for (cluster_level in unique(clusters)){
   batch_tmp <- batch[clusters == cluster_level]
   data_tmp <- data[clusters == cluster_level,]
   kBET_tmp <- kBET(df=data_tmp, batch=batch_tmp, plot=FALSE)
   kBET_result_list[[cluster_level]] <- kBET_tmp
   #ascore <- kBET_tmp$summary$kBET.observed[1]
   ascore <- kBET_tmp$average.pval
   #print(cluster_level)
   #print(ascore)
   dic_kBET_result[cluster_level] <- ascore
   sum_kBET <- sum_kBET + ascore
}

#averaging
mean_kBET = sum_kBET/length(unique(clusters))
