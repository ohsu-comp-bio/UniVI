import numpy as np
import pandas as pd
from collections import defaultdict

from UniVI.utilities._utils import Lists

from sklearn.metrics import mean_squared_error

def calc_rmse(mtx_x1, mtx_x2):
    return mean_squared_error(mtx_x1, mtx_x2, squared=False)

class MapEval:
    
    def __init__(self, df_z1_ident, df_z2_ident):
        
        self.ident_1 = df_z1_ident.ident.tolist()
        self.ident_2 = df_z2_ident.ident.tolist()
        
        self.df_z1 = df_z1_ident.drop(columns=['ident'])
        self.df_z2 = df_z2_ident.drop(columns=['ident'])
    
    def compare(self):
        record_all = list()
        for i in range(self.df_z1.shape[0]):
            df_z1_sample = self.df_z1.iloc[i:i+1,:]
            barcode_query = df_z1_sample.index.values.tolist()
            df_temp_delta = self.get_df_temp_delta_per_sample(df_z1_sample)
            barcode_top_match = df_temp_delta[df_temp_delta['rank'] == 1].index

            rec_self = Lists.flatten(df_temp_delta.loc[barcode_query].values.tolist())
            rec_top_match = Lists.flatten(df_temp_delta.loc[barcode_top_match].values.tolist())
            record_per_sample = [
                *barcode_query,
                *rec_self,
                *rec_top_match
            ]
            record_all.append(record_per_sample)
            
        columns = ["id", "delta_self", "rank", "ident", "id_self", 
                   "delta_top", "rank_top", "ident_top", "id_top"]
        df_compare = pd.DataFrame(record_all, columns=columns)
        df_compare.index = df_compare.id
        df_compare.index.name = None
        df_compare.drop(columns=["id", "id_self", "rank_top"], inplace=True)
        return df_compare
            
    def get_df_temp_delta_per_sample(self, df_sample_query):
        delta = self.calc_delta_one_to_n(df_sample_query.values, self.df_z2.values)
        df_temp_delta = pd.DataFrame(delta, index=self.df_z2.index, columns=["delta"])
        df_temp_delta['rank'] = df_temp_delta.delta.rank()
        df_temp_delta['ident_map'] = self.ident_2
        df_temp_delta['barcode'] = self.df_z2.index
        #df_temp_delta['rank'] = pd.to_numeric(df_temp_delta['rank'], downcast='integer')
        return df_temp_delta
        
    def calc_delta_one_to_n(self, vector_query, mtx):
        return np.sqrt(np.sum(np.square(np.subtract(vector_query, mtx)), axis=1))        

class MVGLikelihood:
    
    def __init__(self, ref_df_ident):
        self.ref_df_ident = ref_df_ident
        self.df_columns = sorted(self.ref_df_ident.ident.unique().tolist())
        self.ref_dic_mvg_stats_ident = self.get_dic_mvg_stats_ident()
        print(f'Cell Types in reference: {self.df_columns}')
        
    @classmethod
    def mvg_likelihood(cls, x, mean, cov):
        x = x.reshape(-1,1)
        mean = mean.reshape(-1,1)
        p = cov.shape[0]

        cov_inv = np.linalg.inv(cov)
        denominator = np.sqrt((2 * np.pi)**p * np.linalg.det(cov))
        exponent = -(1/2) * ((x - mean).T @ cov_inv @ (x - mean))

        return float((1. / denominator) * np.exp(exponent) )

    def predict_celltype_from_z(self, df_z):
        dim_z = self.ref_df_ident.shape[1] - 1
        assert df_z.shape[1] == dim_z, f'should have the same dimension with reference: {dim_z}'
        llik = df_z.apply(lambda x: self.get_llik_per_sample_ident(x.values), axis=1)
        df_llik_ident = pd.DataFrame(list(llik.values), index=df_z.index, columns=self.df_columns)
        df_llik_ident['predict'] = df_llik_ident.idxmax(axis=1)
        return df_llik_ident        

    def get_llik_per_sample_ident(self, x):
        
        lst_llik_ident = list()
        for key_ident in self.df_columns:
            mean = self.ref_dic_mvg_stats_ident[key_ident]['mean']
            cov = self.ref_dic_mvg_stats_ident[key_ident]['cov']
            
            assert x.shape[0] == mean.shape[0] == cov.shape[0]
            
            log_lik = np.log(MVGLikelihood.mvg_likelihood(x,mean,cov))
            lst_llik_ident.append(log_lik)

        return lst_llik_ident        
        
    def get_dic_mvg_stats_ident(self):
        ''' could be used for both df_zs_ident and df_es_ident
        should have a field of 'ident'
        '''
        df_mean = self.ref_df_ident.groupby('ident').mean()
        df_cov = self.ref_df_ident.groupby('ident').cov()

        ref_dic_mvg_stats = defaultdict(dict)
        for key_ident in self.df_columns:
            ref_dic_mvg_stats[key_ident]['mean'] = df_mean.loc[key_ident].values
            ref_dic_mvg_stats[key_ident]['cov'] = df_cov.loc[key_ident].values

        return ref_dic_mvg_stats 
