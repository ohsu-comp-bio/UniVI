import scanpy as sc
from pathlib import Path
import anndata
import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt

from scvt.utilities._evaluate import EvalMap, predict_celltype

# summary file boxplot

def summary():
    
    infiles = glob.glob('/home/groups/precepts/chhy/lib_samples/integration/ds_cit_hao/eval_clustering/summary_synomicvae_*.csv')

    lst_imgs = []
    lst_titles = []
    for i, infile in enumerate(infiles):
        # exp_name
        exp_name = Path(infile).name
        exp_name = exp_name.split('summary_synomicvae_')[1]
        exp_name = exp_name.split('.csv')[0]

        df = pd.read_csv(infile, index_col=0)
        df.boxplot()
        plt.ylim((0,1))
        name_box = 'box_' + str(i+1) + '.png'
        plt.savefig(name_box)
        plt.close()
        lst_imgs.append(name_box)
        lst_titles.append(exp_name)

    grid_display(lst_imgs, ncols=5, lst_titles=lst_titles)        
    return

# hao_sample
'''
# evaluate hao_samples 

hao_samples = glob.glob('/home/groups/precepts/chhy/kci/kci_20200321_multi_modal_vae/vrun_dir/*/hao_sample_*')

# draw
for save_path in hao_samples:
    evalhao(save_path)

# diplay    
lst_imgs = []
lst_titles = []
for save_path in hao_samples:
    file_fig_merged = save_path + '/anal/plot_eval_amerge.png'
    lst_imgs.append(file_fig_merged)
    lst_titles.append(save_path)
    
grid_display(lst_imgs, lst_titles=lst_titles, ncols=2)

save_path = '/home/groups/precepts/chhy/kci/kci_20200321_multi_modal_vae/vrun_dir/beta60/hao_sample_500_p20_multi_gaussian_z50_l5_l5_m_elbo_mc20_b200/'


def evalhao(save_path):
    
    save_path = save_path + '/'
    data_id = 'hao_sample'
    tool_id = 'synomicvae'

    eval_file = save_path + 'anal/eval_' + tool_id + '_' + data_id + '.h5ad'
    aint_file = save_path + 'anal/aint_' + tool_id + '_' + data_id + '.h5ad'
    
    savepath_sankey = save_path + 'anal/plot_eval_sankey' + tool_id + '_' + data_id + '.png'
    
    save_name_umap = '_tmp.png'
    path_umap = 'figures/umap' + save_name_umap

    file_fig_merged = save_path + 'anal/plot_eval_amerge.png'
    
    if Path(file_fig_merged).exists():
        return
    
    if Path(aint_file).exists():
        aint = anndata.read(aint_file)
    else:
        try:
            aint = create_comb_anndata_to_eval(eval_file, outpath = aint_file)
        except:
            raise ValueError(f"not exist: {eval_file}")
       
    if not Path(savepath_sankey).exists():
        predict_celltype(aint)
        EvalMap(aint, show_plot=False, savepath=aint_file, savepath_sankey=savepath_sankey).all()

    # umap
    color = ['pred_kmeans','ident']
    sc.pl.umap(aint, color=color, ncols=4, show=False, save=save_name_umap)

    # merged figure
    grid_display([path_umap, savepath_sankey], ncols=2, filepath_out=file_fig_merged, figratio=1.0)

    plt.close('all')
    return 

'''

# hao20
'''

batch_size = 500
n_latent = 50
n_layer_1 = 5
n_layer_2 = 5
beta = 200
exp = 'sample_denom'

evalbp(exp, batch_size, n_latent, n_layer_1, n_layer_2, beta)

def evalbp(exp, batch_size, z, l1, l2, beta):
    # for summary file naming
    condition = f"{exp}_{batch_size}_z{z}_l{l1}_l{l2}_b{beta}"

    lst_nmi = []
    lst_ari = []
    lst_acc_clu = []
    lst_sil = []

    for i in range(20):

        data_id = 'hao20_' + str(i+1)
        save_path = '/home/groups/precepts/chhy/kci/kci_20200321_multi_modal_vae/vrun_dir/' + exp + '/'

        dir_name = data_id + f"_{batch_size}_p20_multi_gaussian_z{z}_l{l1}_l{l2}_m_elbo_mc20_b{beta}"
        eval_file = save_path + dir_name + '/anal/eval_synomicvae_' + data_id + '.h5ad'
        aint_file = save_path + dir_name + '/anal/aint_synomicvae_' + data_id + '.h5ad'

        if Path(aint_file).exists():
            aint = anndata.read(aint_file)
        else:
            try:
                aint = create_comb_anndata_to_eval(eval_file, outpath = aint_file)
            except:
                print(f"not exist: {aint_file}")
                break
            predict_celltype(aint)
            EvalMap(aint, show_plot=False, savepath=aint_file).all()

        lst_nmi.append(aint.uns['nmi'])
        lst_ari.append(aint.uns['ari'])
        lst_acc_clu.append(aint.uns['acc_clu'])
        lst_sil.append(aint.uns['sil'])    

    df_eval = pd.DataFrame(zip(lst_nmi, lst_ari, lst_acc_clu, lst_sil))    
    df_eval.columns = ["nmi","ari","acc_clu","sil"]

    tool_id = 'synomicvae'
    save_path = '/home/groups/precepts/chhy/lib_samples/integration/ds_cit_hao/'
    file_summary = save_path + 'eval_clustering/summary_' + tool_id + '_' + condition + '.csv'
    df_eval.to_csv(file_summary)

    df_eval.boxplot()
    plt.ylim(0,1)
    print(file_summary)
    df_eval.head()
    return df_eval
'''


def zadata(adata_ref, key_obsm='pred_z'):
    mtx = adata_ref.obsm[key_obsm]
    adata = anndata.AnnData(mtx)
    adata.obs = adata_ref.obs
    
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    return adata

def sub_ad(adata, lst_categories, key_ident='ident'):
    adata_sub = []
    for category in lst_categories:
        adata_sub.append(adata[adata.obs[key_ident] == category])

    adata_out = anndata.concat(adata_sub)
    adata_out.obs[key_ident] = pd.Categorical(adata_out.obs[key_ident])
    
    return adata_out

def umap_ad(adata, color=['ident']):
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=color)


def evalmap(aint, sid=None, key_ref='ident', show_plot=True):
    
    predict_celltype(aint, key_ref=key_ref)
    
    print(f'key_ref: {key_ref}')
    
    # 'savepath=aint.uns['savepath']' added by Andrew 3/28/2023 because I was getting an error when trying to run evalmap:
    EvalMap(aint, savepath=aint.uns['savepath'], key_cluster=key_ref).all()
    
    if sid == 'hao':
        #color = ['init_kmeans','pred_kmeans','ident','level2','level3','batch','donor','lane']
        #color = ['init_kmeans','batch','donor','level3','pred_kmeans','ident','level2']
        color = ['pred_kmeans','ident','level2']
    elif sid == 'PBMC_multi_omics':
        color = ['domain', 'protocol', 'dataset', 'pred_kmeans', 'ident']
    else:
        color = ['batch','pred_init','pred_kmeans','ident']
    if show_plot:
        sc.pl.umap(aint, color=color, ncols=4)
    

def create_comb_anndata_to_eval(infile1, infile2=None, outpath='aint_saved.h5ad', infile_ref_adata=None, ref_adata_key='ident'):
    infile1 = Path(infile1)
    suffix1 = infile1.suffix
    
    if suffix1 == '.csv':
        assert infile_ref_adata is not None, f"infile_ref_adata is required for cell type info"
        ref_adata = anndata.read(infile_ref_adata)
        
    # if we have one input file
    if infile2 is None: # only one infile
        
        if suffix1 == '.h5ad':  # assume to be already combined anndata with 'batch' info
            adata_combined = anndata.read(infile1)
            print(f"- assumming to be an already combined anndata")
            print(f"- checking slots for 'batch' and 'ident'")
            
            try:
                batch = adata_combined.obs['batch']
            except KeyError:
                print(f"- setting 'batch' to all '1'")           
                adata_combined.obs['batch'] = pd.Categorical([1] * adata_combined.shape[0])

            assert adata_combined.obs['ident'] is not None
            print(f"- umap running overriden obsm['umap'] slot")
            
            sc.pp.neighbors(adata_combined)#, use_rep='X')
            sc.tl.umap(adata_combined)
            #adata_combined.uns['savepath'] = outpath
            adata_combined.write(outpath)
            return adata_combined            

        elif suffix1 == '.csv':
            df1 = pd.read_csv(infile1)
            
            if df1.shape[1] == 3:  # with ident info 2D
                print('with batch info 2D csv... but later')
            
            elif df1.shape[1] == 2:  # no ident info 2D for seurat
                print(f"- '{infile1}' file with no ident info")
                batch = [1] * df1.shape[0]
                adata = anndata.AnnData(df1)
                assert all(adata.obs.index == ref_adata.obs.index), \
                        f"should have the common barcode IDs: {infile1}, {infile_ref_adata}"
                print(f"- add batch label '1'")
                adata.obs = ref_adata.obs
                adata.obs['batch'] = pd.Categorical(batch)
                adata.obsm['umap'] = adata.X 
                print(f"- add .X data as umap coordinates to obsm['umap'] slot")
                
                #adata.uns['savepath'] = outpath
                adata.write(outpath)
                return adata
            
            elif df1.shape[1] > 3:  # no batch info N-D
                print('no batch info N-D .csv.. but later')
        else:
            raise ValueError(f"Invalid input: {infile1}")
    
    # if we have two input files
    else:
        
        infile2 = Path(infile2)
        suffix2 = infile2.suffix        
        
        assert suffix1 == suffix2, "should be the same format"

        if suffix1 == '.csv':
            df1 = pd.read_csv(infile1)
            df2 = pd.read_csv(infile2)
            assert df1.shape[1] == df2.shape[1], "should be the same dimension"
            adata1 = anndata.AnnData(df1)
            adata2 = anndata.AnnData(df2)
            
        elif suffix1 == '.h5ad':
            print(f"- read in two .h5ad: {infile1}, {infile2}")
            adata1 = anndata.read(infile1)
            adata2 = anndata.read(infile2)
            
        else:
            print(f"Invalid input file {infile1}, {infile2}")

        assert all(adata1.obs.index == adata2.obs.index), f"should have the common barcode IDs: {infile1}, {infile2}"
        assert adata1.shape[1] == adata2.shape[1], "should be the same dimension"
        
        # if umap coords, do not run umap again
        if adata1.shape[1] == 2:
            #######adata_combined = adata1.concatenate(adata2)
            #adata_combined = adata1.concat(adata2, join="inner")
            #adata_combined = adata1.concat(adata2, join="outer")
            #adata_combined = adata1.concatenate(adata2, join="inner")
            
            # Hyeyoung's code concatenates the aint files like so: aint = a1.concatenate(a2)
            #adata_combined = adata1.concatenate(adata2, fill_value=0).obs # join="outer")
            adata_combined = adata1.concatenate(adata2, fill_value=0) # join="outer")
            #adata_combined = adata1.merge(adata2)
            
            adata_combined.obsm['umap'] = adata_combined.X
            
            print(f"- add .X data as umap coordinates to obsm['umap'] slot")            

        # if N-dim latent features
        elif adata1.shape[1] > 2:
            # preprocessing and runnning for umap
            ######adata_combined = adata1.concatenate(adata2)
            #adata_combined = adata1.concat(adata2, join="inner")
            #adata_combined = adata1.concat(adata2, join="outer")
            #adata_combined = adata1.concatenate(adata2, join="inner")
            #adata_combined = adata1.concatenate(adata2, join="outer", fill_value=0)
            #adata_combined = adata1.merge(adata2)
            
            # Hyeyoung's code concatenates the aint files like so: aint = a1.concatenate(a2)
            #adata_combined = adata1.concatenate(adata2, fill_value=0).obs # join="outer")
            adata_combined = adata1.concatenate(adata2, fill_value=0) # join="outer")
            
            sc.pp.neighbors(adata_combined, use_rep='X')
            sc.tl.umap(adata_combined)
        
        #adata_combined.uns['savepath'] = outpath
        
        adata_combined.write(outpath)
        return adata_combined





