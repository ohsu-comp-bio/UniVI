from UniVI.utilities._utils import str_to_path, Lists, check_mtx_to_df
from typing import Union
from pathlib import Path
import os

import logging
#from anndata import read_csv
import anndata
import scanpy as sc
import numpy as np

from UniVI.utilities._utils import Lists

logger = logging.getLogger(__name__)

def subsample_anndata(adata_ref, N=None, skip_lt_N=True, key_category='ident', verbose=False, seed=42):
    ''' return anndata
    '''
    np.random.seed(seed)

    if N is None:
        N = np.min(adata_ref.obs[key_category].value_counts())

    lst_ids_sampled = []
    if verbose:
        print('[ Original anndata: ]')
    for ident in adata_ref.obs[key_category].cat.categories:
        
        ad_part = adata_ref[adata_ref.obs[key_category] == ident,]
        if verbose:
            print(ident, ad_part.shape)
        ids_shuffled = list(np.random.permutation(ad_part.obs.index))
        cluster_size = len(ids_shuffled)
        
        if cluster_size < N:
            if verbose:
                print(f'\t- the cluster size is less than {N}')
            if skip_lt_N:
                if verbose:
                    print(f'\t=> just skip the cluster {ident}')
                continue
            ids_shuffled = ids_shuffled
        else:
            ids_shuffled = ids_shuffled[:N]
            
        lst_ids_sampled = lst_ids_sampled + ids_shuffled

    # shuffle again the final list of barcode ids    
    lst_ids_sampled = list(np.random.permutation(lst_ids_sampled))

    return adata_ref[lst_ids_sampled,:]

def split_anndata(adata_comb, batch_key='batch'):
    lst_adatas = []
    for batch in adata_comb.obs[batch_key].cat.categories:
        lst_adatas.append(adata_comb[adata_comb.obs[batch_key] == batch])
    return lst_adatas

def combine_anndata(mtx1, mtx2, ad_ref_1=None, ad_ref_2=None):
    '''
    N = 100
    adp1 = ad1[:N,]
    adp2 = ad2[:N,]
    z1 = Z1.values[:N,]
    z2 = Z2.values[:N,]
    
    aintp = combine_anndata(z1,z2,adp1,adp2)
    sc.pl.umap(aintp, color=['batch', 'ident'])
    '''    
    assert mtx1.shape[1] == mtx2.shape[1]
    
    df1 = check_mtx_to_df(mtx1)
    df2 = check_mtx_to_df(mtx2)

    adata1 = anndata.AnnData(mtx1)
    adata2 = anndata.AnnData(mtx2)
    
    if ( ad_ref_1 is not None ) and ( ad_ref_2 is not None ):
        adata1.obs = ad_ref_1.obs
        adata2.obs = ad_ref_2.obs
    
    adata_comb = adata1.concatenate(adata2)
    
    sc.tl.pca(adata_comb)
    sc.pp.neighbors(adata_comb)
    sc.tl.umap(adata_comb)
    return adata_comb    

def _read_csv_to_adata(
    filepath: Union[str, os.PathLike]
) -> anndata._core.anndata.AnnData:
    '''
    filepath = 'integration/ds_cit_hao/hao20/hao20_1_rna_raw.csv'
    adata = _read_csv_to_adata(filepath)
    adata
    '''

    filepath = str_to_path(filepath) # to PosixPath

    delimiter = None
    if filepath.suffix == '.csv': 
        delimiter = ","
    elif filepath.suffix == '.tsv': 
        delimiter = "\t"
    else: 
        raise NotImplementedError(f'Unknown input format: {filepath}')

    adata = anndata.read_csv(
            filename=filepath,
            delimiter=delimiter,
            first_column_names=True,
            dtype='float32'
    ) 

    return adata


class PrepAnnDataRNA:
    
    def __init__(self, 
        filepath_in, 
        filepath_out_h5ad='prep.h5ad', 
        save=False,
        skip_filter=False,
        show_plot=True,
        features_to_select=None,
        missing_impute=False,
        cells_threshold_min_genes=200,
        genes_threshold_min_cells=3,
        obs_threshold_pct_counts_mt = 10,
        percent_outlier = 0.5,
        target_sum = 1,
        n_top_genes = 2000,
        scale_max_value = 10
    ):
        ''' outfile path '''
        #self.filepath_out_h5ad = filepath_in.rpartition('.')[0] + '.h5ad'
        self.filepath_out_h5ad = filepath_out_h5ad
        ''' attributes '''
        self.save = save
        self.features_to_select = features_to_select
        self.missing_impute = missing_impute
        self.cells_threshold_min_genes = cells_threshold_min_genes
        self.genes_threshold_min_cells = genes_threshold_min_cells
        # should be less than this percentage
        self.obs_threshold_pct_counts_mt = obs_threshold_pct_counts_mt 
        # consider top and bottom % of cells as doublets or empty cells
        self.percent_outlier = percent_outlier 
        self.target_sum = target_sum
        self.n_top_genes = n_top_genes
        self.scale_max_value = scale_max_value
        
        ''' init anndata '''
        self.adata = None
        self.features_missing = None
        self.features_common = None
        self._infile_to_adata(filepath_in)
        ''' get stats '''
        self._basic_stats()
        ''' filter out cells and genes '''
        if not skip_filter: 
            if self.percent_outlier > 0.:
                self._remove_outliers()
            self._filter_out_cells_genes()
        ''' rawdata backup for all genes before normalization '''
        self.adata.layers['counts']= self.adata.X.copy()
        self._normalize_log1p()
        self._select_genes()
        ''' scaling '''
        self._scale()
        ''' save to file'''
        if self.save:
            self.adata.write(self.filepath_out_h5ad)
        ''' show QC plot '''
        if show_plot:
            self.adata.obs.hist(bins=100, layout=(1,5), figsize=(14,2))
        
    def _infile_to_adata(self, filepath_in):
        filepath_in = Path(filepath_in)
        if filepath_in.suffix == '.csv':
            self.adata = _read_csv_to_adata(filepath_in)
        elif filepath_in.suffix == '.h5ad':
            self.adata = anndata.read_h5ad(filepath_in)
        else: raise ValueError(f"Invalid file format: {filepath_in}")

    def _basic_stats(self):
        print('get statistics')
        # annotate the group of mitochondrial genes as 'mt'
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')  
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], 
				percent_top=None, log1p=False, inplace=True)
        #print(self.adata)

    def _remove_outliers(self):
        print(f'remove outliers: top and bottom {self.percent_outlier/2} %')
        total_counts = self.adata.obs.total_counts.values
        bottom = np.percentile(total_counts, self.percent_outlier)
        top = np.percentile(total_counts, 100-self.percent_outlier)
        is_valid = (total_counts > bottom) & (total_counts < top)
        self.adata = self.adata[is_valid,:]
        #print(self.adata)

    def _filter_out_cells_genes(self):
        print(f'filter out cells and genes with low quality: ')
        print(f'genes<{self.cells_threshold_min_genes}, cells<{self.genes_threshold_min_cells}')
        # filter out cells and genes with low number of invalid values
        sc.pp.filter_cells(self.adata, min_genes=self.cells_threshold_min_genes)
        sc.pp.filter_genes(self.adata, min_cells=self.genes_threshold_min_cells)
        self.adata = self.adata[self.adata.obs.pct_counts_mt < self.obs_threshold_pct_counts_mt, :].copy()
        #print(self.adata)

    def _normalize_log1p(self):
        print('normalize and log1p transformation')
        # library size normalization 
        sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
        sc.pp.log1p(self.adata)
        #print(self.adata)

    def _impute_missing_genes(self):
        print("need to implement: _impute_missing_genes")
        assert False

    def _handle_missing_genes(self):
        assert len(self.features_to_select)>1
        if Lists.exist_all_in_lists(self.features_to_select, self.adata.var.index.tolist()):
            self.adata = self.adata[:, self.features_to_select]
        else:
            self.features_missing = Lists.items_not_in_list(self.features_to_select, 
			   			  self.adata.var.index.tolist())

            self.features_common = Lists.items_in_list(self.features_to_select, 
			   			  self.adata.var.index.tolist())
            if self.missing_impute:
                self._impute_missing_genes()
            else:
                #raise ValueError(f"Invalid gene names in the list: {self.features_missing}")
                print(f'Build anndata only with genes being in the matrix')
                print(f'List of missing genes are stored in uns[\'missing_genes\']')
                print(f'{len(self.features_missing)}')
                self.adata = self.adata[:, self.features_common]
                self.adata.uns['missing_genes'] = self.features_missing

    def _select_genes(self):
        print('select genes')
        if self.features_to_select is not None:
            print(f'by list of genes: {len(self.features_to_select)}')
            self._handle_missing_genes()
        else:
            print(f'by highly variable genes criteria: {self.n_top_genes}')
            #sc.pp.highly_variable_genes(self.adata, flavor='seurat', n_top_genes=self.n_top_genes)

            sc.pp.highly_variable_genes(self.adata, 
    					n_top_genes=self.n_top_genes,
    					subset=True,
    					layer="counts",
    					flavor="seurat_v3")

            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
            assert self.adata.shape[1] == self.n_top_genes, 'something wrong in the feature dim.'
        #print(self.adata)        

    def _scale(self):
        self.adata.layers['lib_normed_log1p'] = self.adata.X.copy()
        #sc.pp.regress_out(self.adata, ['donor'])
        sc.pp.scale(self.adata, max_value=self.scale_max_value)        
        self.adata.layers['z_scaled'] = self.adata.X.copy()
        #print(self.adata)
        

class PrepAnnDataADT:

    def __init__(self,
        filepath_in,
        filepath_out_h5ad='prep.h5ad', 
        save=False,
        skip_filter=False,
        show_plot=True,
        features_to_select=None,
        missing_impute=False,
        percent_outlier = 0.5,
        target_sum = 1,
        scale_max_value = 10
    ):
        ''' outfile path '''
        #self.filepath_out_h5ad = filepath_in.rpartition('.')[0] + '.h5ad'
        self.filepath_out_h5ad = filepath_out_h5ad
        ''' attributes '''
        self.save = save
        self.features_to_select = features_to_select
        self.missing_impute = missing_impute
        # consider top and bottom % of cells as doublets or empty cells
        self.percent_outlier = percent_outlier
        self.target_sum = target_sum
        self.scale_max_value = scale_max_value

        ''' init anndata '''
        self.adata = None
        self.features_missing = None
        self.features_common = None
        self._infile_to_adata(filepath_in)
        ''' get stats '''
        self._basic_stats()


        ''' filter out cells and genes '''
        if not skip_filter:
            if self.percent_outlier > 0.:
                self._remove_outliers()
        ''' rawdata backup for all genes before normalization '''
        self.adata.layers['counts']= self.adata.X.copy()
        self._normalize_log1p()
        self._select_genes()
        ''' scaling '''
        self._scale()
        ''' save to file'''
        if self.save:
            self.adata.write(self.filepath_out_h5ad)
        ''' show QC plot '''
        if show_plot:
            self.adata.obs.hist(bins=100, layout=(1,5), figsize=(14,2))

    def _infile_to_adata(self, filepath_in):
        filepath_in = Path(filepath_in)
        if filepath_in.suffix == '.csv':
            self.adata = _read_csv_to_adata(filepath_in)
        elif filepath_in.suffix == '.h5ad':
            self.adata = anndata.read_h5ad(filepath_in)
        else:
            raise ValueError(f"Invalid file format: {filepath_in}")

    def _basic_stats(self):
        print('get statistics')
        # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.adata,percent_top=None, log1p=False, inplace=True)
        #print(self.adata)

    def _remove_outliers(self):
        print(f'remove outliers: top and bottom {self.percent_outlier/2} %')
        total_counts = self.adata.obs.total_counts.values
        bottom = np.percentile(total_counts, self.percent_outlier)
        top = np.percentile(total_counts, 100-self.percent_outlier)
        is_valid = (total_counts > bottom) & (total_counts < top)
        self.adata = self.adata[is_valid,:]
        #print(self.adata)

    def _normalize_log1p(self):
        print('normalize and log1p transformation')
        # library size normalization 
        sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
        sc.pp.log1p(self.adata)
        #print(self.adata)

    def _impute_missing_genes(self):
        print("need to implement: _impute_missing_genes")
        assert False

    def _handle_missing_genes(self):
        assert len(self.features_to_select)>1
        if Lists.exist_all_in_lists(self.features_to_select, self.adata.var.index.tolist()):
            self.adata = self.adata[:, self.features_to_select]
        else:
            self.features_missing = Lists.items_not_in_list(self.features_to_select,
                                                  self.adata.var.index.tolist())

            self.features_common = Lists.items_in_list(self.features_to_select,
                                                  self.adata.var.index.tolist())
            if self.missing_impute:
                self._impute_missing_genes()
            else:
                #raise ValueError(f"Invalid gene names in the list: {self.features_missing}")
                print(f'Build anndata only with genes being in the matrix')
                print(f'List of missing genes are stored in uns[\'missing_genes\']')
                print(f'{len(self.features_missing)}')
                self.adata = self.adata[:, self.features_common]
                self.adata.uns['missing_genes'] = self.features_missing

    def _select_genes(self):
        print('select genes')
        if self.features_to_select is not None:
            print(f'by list of genes: {len(self.features_to_select)}')
            self._handle_missing_genes()
        else:
            print(f'by using all features')
        #print(self.adata)

    def _scale(self):
        self.adata.layers['lib_normed_log1p'] = self.adata.X.copy()
        #sc.pp.regress_out(self.adata, ['donor'])
        sc.pp.scale(self.adata, max_value=self.scale_max_value)
        self.adata.layers['z_scaled'] = self.adata.X.copy()
        #print(self.adata)


class PrepAnnDataATAC:
    
    def __init__(self, 
        filepath_in, 
        filepath_out_h5ad='prep_atac.h5ad', 
        save=False,
        skip_filter=False,
        show_plot=True,
        features_to_select=None,
        missing_impute=False,
        cells_threshold_min_peaks=500,
        peaks_threshold_min_cells=5,
        obs_threshold_pct_counts_mt = 10,
        percent_outlier = 0.5,
        target_sum = 1,
        n_top_peaks = 2000,
        scale_max_value = 10
    ):
        ''' outfile path '''
        self.filepath_out_h5ad = filepath_out_h5ad
        ''' attributes '''
        self.save = save
        self.features_to_select = features_to_select
        self.missing_impute = missing_impute
        self.cells_threshold_min_peaks = cells_threshold_min_peaks
        self.peaks_threshold_min_cells = peaks_threshold_min_cells
        self.obs_threshold_pct_counts_mt = obs_threshold_pct_counts_mt 
        self.percent_outlier = percent_outlier 
        self.target_sum = target_sum
        self.n_top_peaks = n_top_peaks
        self.scale_max_value = scale_max_value
        
        ''' init anndata '''
        self.adata = None
        self.features_missing = None
        self.features_common = None
        self._infile_to_adata(filepath_in)
        ''' get stats '''
        self._basic_stats()
        ''' filter out cells and peaks '''
        if not skip_filter: 
            if self.percent_outlier > 0.:
                self._remove_outliers()
            self._filter_out_cells_peaks()
        ''' rawdata backup for all peaks before normalization '''
        self.adata.layers['counts'] = self.adata.X.copy()
        self._normalize_log1p()
        self._select_peaks()
        ''' scaling '''
        self._scale()
        ''' save to file'''
        if self.save:
            self.adata.write(self.filepath_out_h5ad)
        ''' show QC plot '''
        if show_plot:
            self.adata.obs.hist(bins=100, layout=(1,5), figsize=(14,2))
        
    def _infile_to_adata(self, filepath_in):
        filepath_in = Path(filepath_in)
        if filepath_in.suffix == '.csv':
            self.adata = _read_csv_to_adata(filepath_in)
        elif filepath_in.suffix == '.h5ad':
            self.adata = anndata.read_h5ad(filepath_in)
        else: 
            raise ValueError(f"Invalid file format: {filepath_in}")

    def _basic_stats(self):
        print('get statistics')
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], 
                                   percent_top=None, log1p=False, inplace=True)

    def _remove_outliers(self):
        print(f'remove outliers: top and bottom {self.percent_outlier/2} %')
        total_counts = self.adata.obs.total_counts.values
        bottom = np.percentile(total_counts, self.percent_outlier)
        top = np.percentile(total_counts, 100 - self.percent_outlier)
        is_valid = (total_counts > bottom) & (total_counts < top)
        self.adata = self.adata[is_valid, :]

    def _filter_out_cells_peaks(self):
        print(f'filter out cells and peaks with low quality: ')
        print(f'peaks<{self.cells_threshold_min_peaks}, cells<{self.peaks_threshold_min_cells}')
        sc.pp.filter_cells(self.adata, min_genes=self.cells_threshold_min_peaks)
        sc.pp.filter_genes(self.adata, min_cells=self.peaks_threshold_min_cells)
        self.adata = self.adata[self.adata.obs.pct_counts_mt < self.obs_threshold_pct_counts_mt, :].copy()

    def _normalize_log1p(self):
        print('normalize and log1p transformation')
        sc.pp.normalize_total(self.adata, target_sum=self.target_sum)
        sc.pp.log1p(self.adata)

    def _impute_missing_peaks(self):
        print("need to implement: _impute_missing_peaks")
        assert False

    def _handle_missing_peaks(self):
        assert len(self.features_to_select) > 1
        if Lists.exist_all_in_lists(self.features_to_select, self.adata.var.index.tolist()):
            self.adata = self.adata[:, self.features_to_select]
        else:
            self.features_missing = Lists.items_not_in_list(self.features_to_select, 
                                                            self.adata.var.index.tolist())
            self.features_common = Lists.items_in_list(self.features_to_select, 
                                                       self.adata.var.index.tolist())
            if self.missing_impute:
                self._impute_missing_peaks()
            else:
                print(f'Build anndata only with peaks being in the matrix')
                print(f'List of missing peaks are stored in uns[\'missing_peaks\']')
                print(f'{len(self.features_missing)}')
                self.adata = self.adata[:, self.features_common]
                self.adata.uns['missing_peaks'] = self.features_missing

    def _select_peaks(self):
        print('select peaks')
        if self.features_to_select is not None:
            print(f'by list of peaks: {len(self.features_to_select)}')
            self._handle_missing_peaks()
        else:
            print(f'by highly variable peaks criteria: {self.n_top_peaks}')
            sc.pp.highly_variable_genes(self.adata, 
                                        n_top_genes=self.n_top_peaks,
                                        subset=True,
                                        layer="counts",
                                        flavor="seurat_v3")
            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
            assert self.adata.shape[1] == self.n_top_peaks, 'something wrong in the feature dim.'

    def _scale(self):
        self.adata.layers['lib_normed_log1p'] = self.adata.X.copy()
        sc.pp.scale(self.adata, max_value=self.scale_max_value)        
        self.adata.layers['z_scaled'] = self.adata.X.copy()
        

