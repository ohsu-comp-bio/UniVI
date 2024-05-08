import anndata2ri
import rpy2.robjects as ro
import sklearn
import numpy as np
import pandas as pd
from UniVI.plotting.sankey import sankey as plot_sankey
from typing import Literal, Union, List
from UniVI.utilities._utils import Lists


def unsupervised_clustering_accuracy( y: np.ndarray, y_pred: np.ndarray):
    ''' adapted from scvi tools '''
    
    if isinstance(y, list):
        y = np.array(y)
        
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        
    from scipy.optimize import linear_sum_assignment

    """Unsupervised Clustering Accuracy."""
    if len(y_pred) != len(y):
        raise ValueError("len(y_pred) != len(y)")
    u = np.unique(np.concatenate((y, y_pred)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(y_pred, y):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1
    cost_matrix = reward_matrix.max() - reward_matrix
    row_assign, col_assign = linear_sum_assignment(cost_matrix)

    # Construct optimal assignments matrix
    row_assign = row_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    col_assign = col_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    assignments = np.concatenate((row_assign, col_assign), axis=1)

    optimal_reward = reward_matrix[row_assign, col_assign].sum() * 1.0
    return optimal_reward / y_pred.size, assignments


def predict_celltype(adata, key_ref = 'ident'):

    def run_kmeans(x, n_clusters):
        model = sklearn.cluster.KMeans(n_clusters, random_state=42)
        out = model.fit(x)
        return out.predict(x)

    def dic_celltype_code_ref(categories):
        dic_cellcode = dict()
        for i, celltype in enumerate(sorted(categories)):
            dic_cellcode[celltype] = i
        return dic_cellcode

    if adata.shape[1] == 2:
        x = adata.obsm['umap']
    else:
        x = adata.obsm['X_umap']
        
    n_clusters = len(adata.obs[key_ref].cat.categories)

    # kmeans prediction
    pred_init = run_kmeans(x, n_clusters)

    # cell type categories
    categories = adata.obs[key_ref].cat.categories

    # reference dic
    dic_ref_cellcode = dic_celltype_code_ref(categories)
    dic_ref_celltype = dict(zip(dic_ref_cellcode.values(), dic_ref_cellcode.keys()))

    # y_true number 
    y_true = np.array(Lists.list_to_hash_value(adata.obs[key_ref].tolist(), dic_ref_cellcode))

    # do cluster match 
    acc, cluster_match = unsupervised_clustering_accuracy(y_true, pred_init)
    dic_cluster_match = dict(cluster_match)
    #dic_cluster_match = dict(zip(cluster_match[:,1], cluster_match[:,0]))

    # cluster number match of y_pred against y_true
    y_pred = np.array(Lists.list_to_hash_value(pred_init, dic_cluster_match))

    # pred to character
    pred_char_kmeans = np.array(Lists.list_to_hash_value(y_pred, dic_ref_celltype))

    # save to each slot
    adata.obs['y_true'] = pd.Categorical(y_true)
    adata.obs['y_pred'] = pd.Categorical(y_pred)
    adata.obs['init_kmeans'] = pd.Categorical(pred_init)
    adata.obs['pred_kmeans'] = pd.Categorical(pred_char_kmeans)


class EvalMap:

    def __init__(self, adata_latent, key_cluster='ident', show_plot=True, savepath=None, savepath_sankey=None, size_inches=(5,5)):

        self.adata = adata_latent
        self.true = None
        self.pred = None

        self.key_cluster = key_cluster
        self.show_plot = show_plot
        self.savepath_sankey = savepath_sankey
        self.savepath = savepath
        self.trueC = None
        self.predC = None

        self._check_anndata()

    def save(self):
        try:
            self.adata.write(self.savepath)
        except:
            raise ValueError(f"savepath is needed")

    def all(self):
        print('- nmi: ', self.nmi())
        print('- ari: ', self.ari())
        print('- acc_clu: ', self.acc_clu())
        print('- sil: ', self.sil())

        self.save()
        if self.show_plot:
            self.sankey()

        if self.savepath_sankey is not None:
            self.sankey(savepath_sankey=self.savepath_sankey)

    def _check_anndata(self):
        try:
            self.true = self.adata.obs['y_true'].tolist()
            self.pred = self.adata.obs['y_pred'].tolist()

            self.trueC = self.adata.obs[self.key_cluster].tolist()
            self.predC = self.adata.obs['pred_kmeans'].tolist()
        except:
            raise ValueError(f"check slots of obs['y_true'], 'y_pred', 'pred_kmeans'")
        return

    def nmi(self):
        nmi = sklearn.metrics.normalized_mutual_info_score(self.true, self.pred)
        self.adata.uns['nmi'] = nmi
        return nmi

    def ari(self):
        ari = sklearn.metrics.cluster.adjusted_rand_score(self.true, self.pred)
        self.adata.uns['ari'] = ari
        return ari

    def acc_clu(self):
        acc_clu, dic_clu = unsupervised_clustering_accuracy(self.true, self.pred)
        self.adata.uns['acc_clu'] = acc_clu
        return acc_clu

    def sil(self):
        sil = sklearn.metrics.silhouette_score(self.adata.X, self.pred)
        self.adata.uns['sil'] = sil
        return sil

    def sankey(self, savepath_sankey=None):
        # need to modify
        categories = self.adata.obs[self.key_cluster].cat.categories.tolist()
        try:
            key_colors = self.adata.uns[self.key_cluster+'_colors']
            colorDict = dict(zip(categories, key_colors))
        except:
            colorDict = None

        if savepath_sankey is not None:
            plot_sankey(self.trueC, self.predC, colorDict=colorDict, figureName=savepath_sankey, fontsize=14)
        else:
            plot_sankey(self.trueC, self.predC, colorDict=colorDict)



def kBET(matrix, batch, clusters=None, k=5):
    '''
    Calculate kBET-pvalue
    # arguments
        matrix:
        batch:
        clusters:
    # return
    
    '''
    import rpy2.robjects as ro
    import anndata2ri
    ro.r(f'library(kBET)')
    anndata2ri.activate()
    
    # check type of 'batch'
    if isinstance(batch, list):
        batch = np.array(batch)
    elif isinstance(batch, np.array):
        batch = batch
    else:
        raise ValueError(f'should be either list or np.array: {batch}')
    
    if clusters is None:
        ro.globalenv['abatch'] = batch
        ro.globalenv['amatrix'] = matrix
        ro.r(f'out.kBET<-kBET(amatrix, abatch)')
        #score_kBET = ro.r(f'out.kBET$summary$kBET.observed[1]')[0]
        score_kBET = ro.r("out.kBET$average.pval")[0]
        return score_kBET

    # check type of 'clusters'
    if isinstance(clusters, list):
        clusters = np.array(clusters)
    elif isinstance(clusters, np.array):
        clusters = clusters
    else: 
        raise TypeError("should be either list or np.array: {clusters}")
    
    assert len(batch) == len(clusters) == matrix.shape[0]
    dic_kBET_result = dict()
    sum_kBET = 0
    for cluster in set(clusters):
        abatch = batch[clusters == cluster]
        amatrix = matrix[clusters == cluster]
        #continue
        ro.globalenv['abatch'] = abatch
        ro.globalenv['amatrix'] = amatrix
        ro.r(f'out.kBET<-kBET(amatrix, abatch)')
        out_kBET = ro.r(f'out.kBET')

        #score_kBET = ro.r(f'out.kBET$summary$kBET.observed[1]')[0]
        score_kBET = ro.r("out.kBET$average.pval")[0]
        #print(cluster, score_kBET)
        dic_kBET_result[cluster] = score_kBET

    anndata2ri.deactivate()
    #return dic_kBET_result, out_kBET
    return dic_kBET_result



