from typing import Literal, Union
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import anndata
import scipy.sparse as ss


from scvt._settings import DataPath

def get_cell_ident_from_reference(lst_barcodes,
        return_kind: Union[Literal["celltype", "ident"]]="celltype",
        data_id='cit_hao', str_level="level1"
):
    if data_id == 'cit_hao':
        dataset = CIT_HAO()
    elif data_id == 'atac_multi':
        dataset = ATA_MULTI()
    else:
        raise ValueError(f"invalid data_id: {data_id}")

    if return_kind == 'celltype':
        ident = dataset.seurat_celltype_by_barcode(lst_barcodes, str_level)
    elif return_kind == 'ident':
        ''' not finished '''
        ident = dataset.seurat_ident_by_barcode(lst_barcodes, str_level)
    else:
        raise ValueError(f"invalid return_kind: {return_kind}")

    return ident


def get_cell_ident_from_reference_OLD(lst_barcodes, 
	return_kind: Union[Literal["celltype", "ident"]]="celltype", 
	data_id='cit_hao', str_level="level1"
):
    if data_id == 'cit_hao' and return_kind == 'celltype':
        hao = CIT_HAO()
        ident = hao.seurat_celltype_by_barcode(lst_barcodes, str_level)
    elif data_id == 'cit_hao' and return_kind == 'ident':
        ''' not finished '''
        hao = CIT_HAO()
        ident = hao.seurat_ident_by_barcode(lst_barcodes, str_level)
    else:
        raise ValueError(f"invalid data_id: {data_id}")

    return ident


class CIT_HAO(object):

    def __init__(self):

        self.df_obs = None
        self.SCT = None
        self.ADT = None

        self.filepath = Path(DataPath.CIT_HAO)
        self.h5 = h5py.File(self.filepath, 'r')
        self.cell_names = self.h5["/cell.names"][()]

        self.dict_df_ident = dict()
        self.dict_df_celltype = dict()

        self._init_data()
        self._init_dataframe_obs()

    def save_to_anndata_SCT(self, outfile_h5ad = 'ds_cit_hao_all_SCT.h5ad'):
        
        ''' var dataframe '''
        features = self.h5["assays/SCT/features"]
        lst_features = features[()]
        df_var = pd.DataFrame(lst_features, index=lst_features, columns=["var_names"])
        
        ''' data matrix '''
        data = self.h5["assays/SCT/counts/data"]
        indices = self.h5["assays/SCT/counts/indices"]
        indptr = self.h5["assays/SCT/counts/indptr"]
        mtx = ss.csr_matrix((data, indices, indptr))

        self.SCT = anndata.AnnData(X=mtx, obs=self.df_obs, var=df_var)
        self.SCT.write(outfile_h5ad)
        print(f'File created: {outfile_h5ad}')

    def _init_dataframe_obs(self):
        ''' obs dataframe '''
        lst_cell_names = self.cell_names
        lst_celltype_l1 = self.h5['/meta.data/celltype.l1'][()]
        lst_celltype_l2 = self.h5['/meta.data/celltype.l2'][()]
        lst_celltype_l3 = self.h5['/meta.data/celltype.l3'][()]
        donor = self.h5['/meta.data/donor'][()]
        lane = self.h5['/meta.data/lane'][()]

        self.df_obs = pd.DataFrame(zip(
        			  lst_celltype_l1,
        			  lst_celltype_l1,
        			  lst_celltype_l2,
        			  lst_celltype_l3,
        			  donor,
        			  lane),
				index=lst_cell_names, 
			columns=["ident", "level1", "level2", "level3", "donor", "lane"])

    def save_to_anndata_ADT(self, outfile_h5ad="ds_cit_hao_all_ADT.h5ad"):
        
        ''' var dataframe '''
        features = self.h5["assays/ADT/features"]
        lst_features = features[()]
        df_var = pd.DataFrame(lst_features, index=lst_features, columns=["var_names"])
        
        ''' data matrix '''
        data = self.h5["assays/ADT/counts/data"]
        indices = self.h5["assays/ADT/counts/indices"]
        indptr = self.h5["assays/ADT/counts/indptr"]
        mtx = ss.csr_matrix((data, indices, indptr))

        self.ADT = anndata.AnnData(X=mtx, obs=self.df_obs, var=df_var)
        self.ADT.write(outfile_h5ad)
        print(f'File created: {outfile_h5ad}')
        
    def _get_df_categorical(self, lst, column_name, barcodes):
        df = pd.DataFrame(lst, columns=[column_name], index=barcodes)
        df[column_name] = pd.Categorical(df[column_name])
        return df

    def _init_data(self):

        barcodes = self.h5['cell.names'][()]

        ''' df_ident level 3: 58 '''
        lst_ident_l3 = self.h5["active.ident/values"][()]
        #lst_celltype_l3 = self.h5["active.ident/levels"][()]

        lst_celltype_l1 = self.h5['/meta.data/celltype.l1'][()]
        lst_celltype_l2 = self.h5['/meta.data/celltype.l2'][()]
        lst_celltype_l3 = self.h5['/meta.data/celltype.l3'][()]

        self.dict_df_ident['level3'] = self._get_df_categorical(lst_ident_l3, 'ident', barcodes)


        self.dict_df_celltype['level1'] = self._get_df_categorical(lst_celltype_l1, 'celltype', barcodes)
        self.dict_df_celltype['level2'] = self._get_df_categorical(lst_celltype_l2, 'celltype', barcodes)
        self.dict_df_celltype['level3'] = self._get_df_categorical(lst_celltype_l3, 'celltype', barcodes)


    def seurat_ident_by_barcode(self, lst_barcodes):
        return list(self.df_ident.loc[lst_barcodes, 'ident'].values)

    def seurat_celltype_by_ident(self, lst_idents):
        return list(self.df_celltype.loc[lst_idents, 'cell_type'].values)

    def seurat_celltype_by_barcode(self, lst_barcodes, str_level="level1"):
        '''
        example >> 
        hao = CIT_HAO()
        ident = hao.seurat_celltype_by_barcode("level1", list(df_x.index))
        '''
        return list(self.dict_df_celltype[str_level].loc[lst_barcodes, 'celltype'].values)

