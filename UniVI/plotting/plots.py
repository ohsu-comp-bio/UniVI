from typing import Union
from plotnine import *
from plotnine.data import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import torch

from UniVI.utilities._utils import check_mtx_to_df, embed_umap, embed_tsne, embed_pca

def scale_iqr(mtx):
    assert isinstance(mtx, np.ndarray)
    scaler = RobustScaler()
    scaler.fit(mtx)
    scaled = scaler.transform(mtx)
    return scaled

def scale_standard(mtx):
    assert isinstance(mtx, np.ndarray)
    scaler = StandardScaler()
    scaler.fit(mtx)
    scaled = scaler.transform(mtx)
    return scaled

def scale_minmax(mtx):
    assert isinstance(mtx, np.ndarray)
    scaler = MinMaxScaler()
    scaler.fit(mtx)
    scaled = scaler.transform(mtx)
    return scaled

def color_scale_minmax(mtx):
    assert isinstance(mtx, np.ndarray)
    scaled_iqr = scale_iqr(mtx)
    scaled_iqr_minmax = scale_minmax(scaled_iqr)
    return scaled_iqr_minmax

def color_scale_standard(mtx):
    assert isinstance(mtx, np.ndarray)
    scaled_iqr = scale_iqr(mtx)
    scaled_iqr_standard = scale_standard(scaled_iqr)
    return scaled_iqr_standard

def nrow_figuresize(n_features):
    if n_features == 1:
        figure_size=(3,3)
        nrow=1
    elif 2 < n_features < 10:
        figure_size=(8,4)
        nrow=2
    elif n_features <= 10:
        figure_size=(20,2)
        nrow=1
    elif 10 < n_features <= 20:
        figure_size=(18,4)
        nrow=2
    elif 20 < n_features <= 30:
        figure_size=(18,4)
        nrow=3
    elif 30 < n_features <= 40:
        figure_size=(18,4)
        nrow=2
    elif 40 < n_features <= 50:
        figure_size=(20,4)
        nrow=3
    else:
        figure_size=(20,4)
        nrow=4
    return nrow, figure_size

def gg_point_feature_active(
    mtx_embed: np.ndarray,
    mtx_feature: np.ndarray,
    title="",
    color='red',
    figure_size = (3,3)
):
    assert mtx_embed.shape[0] == mtx_feature.shape[0]
    assert mtx_embed.shape[1] == 2 # two-dimensional coordinates
    assert mtx_feature.shape[1] == 1 # one-dimensional coordinates
    assert isinstance(mtx_embed, np.ndarray)
    assert isinstance(mtx_feature, np.ndarray)

    df_concat = pd.DataFrame(np.concatenate([mtx_embed, mtx_feature], axis=1))
    df_concat.columns = ['x','y','score']

    return ggplot() +\
        geom_point(df_concat, aes(x="x", y="y", color="score"), size=0.01, alpha=1.0, show_legend=False) +\
        scale_color_gradient(low="yellow", high=color) +\
        theme_bw() +\
        theme(
                    figure_size=figure_size,
                    axis_text=element_blank(),
                    axis_ticks=element_blank(),
                    panel_grid=element_blank(),
                    legend_position=None,
                    legend_text=element_blank(),
                    legend_entry_spacing=element_blank(),
                    legend_title=element_blank(),
                    legend_background=element_blank(),
                    legend_key = element_blank(),
                    text = element_text(size=9)
        ) +\
        labs(x="", y="", title=title)


def gg_point_feature_active_to_be_removed(
    df_cell_embed: pd.DataFrame,
    df_feature_data: pd.DataFrame,
    title='',
    color='red'
):
    assert all(df_cell_embed.index == df_feature_data.index)
    assert df_cell_embed.shape[1] == 2 # two-dimensional coordinates
    assert isinstance(df_cell_embed, pd.DataFrame)
    assert isinstance(df_feature_data, pd.DataFrame)
    
    nrow, figure_size = nrow_figuresize(df_feature_data.shape[1])

    '''
    df_feature_data_scaled = pd.DataFrame(color_scale_minmax(df_feature_data.values))
    df_feature_data_scaled.index = df_feature_data.index
    df_feature_data_scaled.columns = df_feature_data.columns
    '''
    df_feature_data_scaled = df_feature_data

    df_cell_embed.columns = ["dim1", "dim2"]

    df_concat = pd.concat([df_cell_embed, df_feature_data_scaled], axis=1)
    df_concat['id'] = df_concat.index

    melted = df_concat.melt(id_vars=["id", "dim1", "dim2"], var_name=["feature"], value_name="score")
    melted['feature'] = pd.Categorical(melted['feature'], categories=df_feature_data.columns)

    return ggplot() +\
        geom_point(df_cell_embed, aes(x="dim1", y="dim2"), color='gray', size=1, alpha=0.3) +\
        geom_point(melted, aes(x="dim1", y="dim2", color="score"), size=1, alpha=1.0) +\
        scale_color_gradient(low="white", high=color) +\
        facet_wrap("feature", nrow=nrow) +\
        theme_bw() +\
        theme(
                    figure_size=figure_size,
                    panel_grid=element_blank(),
                    legend_title=element_blank(),
                    legend_background=element_blank(),
                    legend_key = element_blank(),
                    text = element_text(size=9)
                ) +\
        labs(x="", y="", title=title)


def gg_point_z_activation(
    zs: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    title=''
):
    if zs.shape[1] <= 10:
        figure_size=(20,2)
        nrow=1
    elif 10 < zs.shape[1] <= 20:
        figure_size=(18,4)
        nrow=2
    elif 20 < zs.shape[1] <= 30:
        figure_size=(18,4)
        nrow=3
    elif 30 < zs.shape[1] <= 40:
        figure_size=(18,4)
        nrow=2
    elif 40 < zs.shape[1] <= 50:
        figure_size=(20,4)
        nrow=3

    ''' for zs dataframe '''
    df_zs = check_mtx_to_df(zs)
    df_zs_columns = ['z'+str(i) for i in range(1,df_zs.shape[1]+1)]
    df_zs.columns = df_zs_columns

    ''' for 2D embed dataframe '''
    df_embed = pd.DataFrame(embed_umap(df_zs.values))
    df_embed.columns = ['dim'+str(i) for i in range(1,df_embed.shape[1]+1)]
    df_embed.index = df_zs.index

    ''' scale for plotting '''
    df_zs_scaled = pd.DataFrame(color_scale_minmax(df_zs.values))
    df_zs_scaled.index = df_zs.index
    df_zs_scaled.columns = df_zs.columns

    ''' for concatenated dataframe '''
    df_concat = pd.concat([df_embed, df_zs_scaled], axis=1)
    df_concat['id'] = df_concat.index

    melted = df_concat.melt(id_vars=["id", "dim1", "dim2"], var_name=["z"], value_name="score")
    melted['z'] = pd.Categorical(melted['z'], categories=df_zs_columns)

    return ggplot() +\
        geom_point(melted, aes(x="dim1", y="dim2", color="score"), size=1, alpha=0.5) +\
        scale_color_gradient(low="white", high="black") +\
        facet_wrap("z", nrow=nrow) +\
        theme_bw() +\
        theme(
                    figure_size=figure_size,
            	    panel_grid=element_blank(),
                    legend_title=element_blank(),
                    legend_background=element_blank(),
                    legend_key = element_blank(),
                    text = element_text(size=9)
                ) +\
        scale_fill_brewer(palette=6) +\
        labs(x="", y="", title=title)


def boxplot_from_mtx(
        mtx: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        xlab = "var", 
        ylab = "score", 
        title = "", 
        figure_size = (4, 3)
):  
    df = check_mtx_to_df(mtx)

    ''' variable names ''' 
    id_cat = df.columns
    df['sid'] = list(df.index)

    ''' melted for ggplot ''' 
    melted = df.melt(id_vars=['sid'], var_name=['var'], value_name="score")
    melted['var'] = pd.Categorical(melted['var'], ordered=True, categories=id_cat)

    return  ggplot (melted) +\
        geom_boxplot(aes(x='var', y='score')) +\
        theme_bw() +\
        theme(
                panel_grid=element_blank(), 
		figure_size=figure_size
        ) +\
        labs(x=xlab, y=ylab, title=title)


def gg_point_pair_embed(
    embed1: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    embed2: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    show_text = False,
    figure_size = (12,12),
    lbl_mod1 = "M1",
    lbl_mod2 = "M2",
    title=""
):
    assert type(embed1) == type(embed2), "should be of the same type"
    df_e1 = check_mtx_to_df(embed1)
    df_e2 = check_mtx_to_df(embed2)
    
    df_e1['M'] = ["M1"] * df_e1.shape[0]
    df_e2['M'] = ["M2"] * df_e2.shape[0]
    df_es = df_e1.append(df_e2)
    df_es.columns = ['dim1','dim2','M']

    ''' new data frame for geom_segment '''
    d = dict()
    d['x'] = df_es[df_es["M"] == 'M1'].dim1.values
    d['y'] = df_es[df_es["M"] == 'M1'].dim2.values
    d['vx'] = df_es[df_es["M"] == 'M2'].dim1.values
    d['vy'] = df_es[df_es["M"] == 'M2'].dim2.values
    df_segment = pd.DataFrame(d)

    ''' new data frame for geom_text '''
    df_text = df_es.copy()

    n_cells = int(df_text.shape[0] / 2)
    cell_label = []
    cell_label.extend(list(range(n_cells)))
    cell_label.extend(list(range(n_cells)))
    df_text["cell_label"] = cell_label

    ''' new data frame for geom_point '''
    df_point = df_text.copy()

    ''' position shift '''
    df_text["dim1"] = df_text["dim1"] + 0.1

    ''' drawing ggplot, position="jitter" '''
    point_label=list(df_text.index)

    if show_text:
        alpha_point = 0.1
    else:
        alpha_point = 0.8

    g = ggplot() +\
           geom_point(data=df_point, mapping=aes(x="dim1", y="dim2", color="M"), position="jitter", alpha=alpha_point) +\
           scale_color_manual(values=["red", "blue"], labels=[lbl_mod1, lbl_mod2]) +\
           geom_segment(data=df_segment, mapping=aes(x="x", y="y", xend="vx", yend="vy"), color="grey", linetype="dashed", alpha=0.3) +\
           theme_bw() +\
           theme(figure_size=figure_size, 
                 panel_grid=element_blank(), 
                 legend_position="bottom", 
                 legend_background=element_blank(),
                 legend_title=element_blank(),
                 legend_key = element_blank(),
                 text = element_text(size=10)
            ) +\
            labs(x="", y="", title=title)

    if show_text:
        g = g + geom_text(data=df_text, mapping=aes(x="dim1", y="dim2", label=cell_label, color="M"), alpha=0.9, size=9)

    return g


def gg_point_pair_by_umap(
    mtx1: Union[pd.DataFrame, np.ndarray, torch.Tensor], 
    mtx2: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    figure_size = (4,4),
    lbl_mod1 = "M1",
    lbl_mod2 = "M2",
    title=""
):  
    assert type(mtx1) == type(mtx2), "should be of the same type"
    
    df1 = check_mtx_to_df(mtx1)
    df2 = check_mtx_to_df(mtx2)

    # batch ID
    batch = ['z1']*df1.shape[0] + ['z2']*df2.shape[0]

    # embed using UMAP
    df_z = pd.concat([df1,df2])
    embeded = embed_umap(df_z)
    df_embeded = pd.DataFrame(embeded, index=df_z.index, columns=['dim1','dim2'])
    df_embeded['batch'] = pd.Categorical(batch)

    # dataframe for segment
    d = dict()
    d['x'] = df_embeded[df_embeded["batch"] == "z1"].dim1.values
    d['y'] = df_embeded[df_embeded["batch"] == "z1"].dim2.values
    d['vx'] = df_embeded[df_embeded["batch"] == "z2"].dim1.values
    d['vy'] = df_embeded[df_embeded["batch"] == "z2"].dim2.values
    df_geom = pd.DataFrame(d)

    # do ggplot
    return ggplot() +\
            geom_point(data=df_embeded, mapping=aes(x="dim1",y="dim2",color='batch'),
                                    size=1, position="jitter", alpha=0.7) +\
            scale_color_manual(values=["red", "blue"], 
                               name="Modality",labels=[lbl_mod1, lbl_mod2]) +\
            geom_segment(data=df_geom,
                        mapping=aes(x="x", y="y", xend="vx", yend="vy"),
                        color="grey", linetype="dashed", alpha=0.5) +\
            theme_bw() +\
            theme(
                    figure_size=figure_size,
                     panel_grid=element_blank(),
                     #legend_position="bottom", 
                     legend_background=element_blank(),
                     legend_title=element_blank(),
                     legend_key = element_blank(),
                     text = element_text(size=8)
            ) +\
            labs(title=title, x="Dim1", y="Dim2")



def gg_point_embed_pair_OLD(
    mtx1: Union[pd.DataFrame, np.ndarray, torch.Tensor], 
    mtx2: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    figure_size = (4,4),
    lbl_mod1 = "M1",
    lbl_mod2 = "M2",
    title=""
):  
    #lbl_mod1, lbl_mod2 = "z1", "z2"
    
    assert type(mtx1) == type(mtx2), "should be of the same type"
    
    df1 = check_mtx_to_df(mtx1)
    df2 = check_mtx_to_df(mtx2)

    ''' merged data frame for geom_point '''
    df = pd.DataFrame(np.concatenate([df1.values, df2.values]), columns=["x", "y"])
    label = [*["z1"]*df1.shape[0], *["z2"]*df2.shape[0]]
    df['label'] = pd.Categorical(label, categories=["z1", "z2"], ordered=True)

    ''' new data frame for geom_segment '''
    d = dict()
    d['x'] = df[df["label"] == "z1"].x.values
    d['y'] = df[df["label"] == "z1"].y.values
    d['vx'] = df[df["label"] == "z2"].x.values
    d['vy'] = df[df["label"] == "z2"].y.values
    df_geom = pd.DataFrame(d)

    ''' drawing ggplot, position="jitter" '''
    return ggplot() +\
           geom_point(data=df, 
			mapping=aes(x="x", y="y", color="label"), 
			position="jitter", alpha=0.7) +\
           scale_color_manual(values=["red", "blue"], name="Modality", 
			labels=[lbl_mod1, lbl_mod2]) +\
           geom_segment(data=df_geom, 
			mapping=aes(x="x", y="y", xend="vx", yend="vy"), 
			color="grey", linetype="dashed", alpha=0.5) +\
           theme_bw() +\
           theme(figure_size=figure_size,
                 panel_grid=element_blank(), 
                 legend_position="bottom", 
                 legend_title=element_blank(),
                 legend_background=element_blank(),
                 legend_key = element_blank(),
                 text = element_text(size=8)
            ) +\
            labs(x="", y="", title=title)


def gg_point_embed(
    mtx, 
    lst_ident=None, 
    categories=None,
    colors=None,
    title='Z-embeded', 
    figure_size=(5,5)
):
    df_embeded = check_mtx_to_df(mtx)
    assert df_embeded.shape[1] == 2, "should be 2D matrix"
    df_embeded.columns =["x", "y"]

    if lst_ident is None:
        #print('here we are')
        return ggplot() +\
            geom_point(data=df_embeded, mapping=aes(x="x",y="y"),
                                    size=2, position="jitter", alpha=0.7) +\
            theme_bw() +\
            theme(
                    figure_size=figure_size,
                     panel_grid=element_blank(),
                     #legend_position="bottom", 
                     legend_background=element_blank(),
                     legend_title=element_blank(),
                     legend_key = element_blank(),
                     text = element_text(size=8)
            ) +\
            labs(title=title, x="Dim1", y="Dim2")

    assert mtx.shape[0] == len(lst_ident), "should be of the same length"
    #df_embeded['ident'] = pd.Categorical(lst_ident, categories=categories, ordered=True) ####???
    df_embeded['ident'] = lst_ident ####??? ????

    ''' ggplot '''
    return gg_point_ident(df_embeded, title=title, figure_size=figure_size, colors=colors)


def gg_point_embed_BAK(
    mtx, lst_ident=None, categories=None, 
    title='Z-embeded', figure_size=(5,5)
):
    df_mtx = check_mtx_to_df(mtx)
    #df_embeded = pd.DataFrame(embed_umap(df_mtx.values), columns=["x", "y"])
    df_embeded = pd.DataFrame(embed_tsne(df_mtx.values), columns=["x", "y"])
    #df_embeded = pd.DataFrame(embed_pca(df_mtx.values), columns=["x", "y"])

    if lst_ident is None:
        #print('here we are')
        return ggplot() +\
            geom_point(data=df_embeded, mapping=aes(x="x",y="y"), 
                                    size=2, position="jitter", alpha=0.7) +\
            theme_bw() +\
            theme(
                    figure_size=figure_size,
                     panel_grid=element_blank(), 
                     #legend_position="bottom", 
                     legend_background=element_blank(),
                     legend_title=element_blank(),
                     legend_key = element_blank(),
                     text = element_text(size=8)
            ) +\
            labs(title=title, x="Dim1", y="Dim2")


    assert mtx.shape[0] == len(lst_ident), "should be of the same length"
    #df_embeded['ident'] = pd.Categorical(lst_ident, categories=categories, ordered=True) ####???
    df_embeded['ident'] = lst_ident ####??? ????

    ''' ggplot '''
    return gg_point_ident(df_embeded, title=title, figure_size=figure_size)

def gg_point_overlay(
    df_target,
    df_ref,
    title='Z-overlayed',
    figure_size=(4,4)
):
    
    assert df_target.shape[1] == 2
    df_target.columns = ["x", "y"]

    if df_ref.shape[1] == 2:
        df_ref.columns = ["x", "y"]
        return ggplot() +\
            geom_point(data=df_ref, mapping=aes(x="x",y="y"), color="gray", 
                                    size=1, position="jitter", alpha=0.7) +\
            geom_point(data=df_target, mapping=aes(x="x",y="y"), color='black',
                                    size=1, position="jitter", alpha=0.7) +\
            theme_bw() +\
            theme(
                    figure_size=figure_size,
                     panel_grid=element_blank(),
                     #legend_position="bottom", 
                     legend_background=element_blank(),
                     legend_title=element_blank(),
                     legend_key = element_blank(),
                     text = element_text(size=8)
            ) +\
            labs(title=title, x="Dim1", y="Dim2")

    elif df_ref.shape[1] == 3:
        df_ref.columns = ["x", "y", "ident"]
        return ggplot() +\
            geom_point(data=df_ref, mapping=aes(x="x",y="y", color="ident"),
                                    size=1, position="jitter", alpha=0.7) +\
            geom_point(data=df_target, mapping=aes(x="x",y="y"), color='black',
                                    size=1, position="jitter", alpha=0.7) +\
            theme_bw() +\
            theme(
                    figure_size=figure_size,
                     panel_grid=element_blank(),
                     #legend_position="bottom", 
                     legend_background=element_blank(),
                     legend_title=element_blank(),
                     legend_key = element_blank(),
                     text = element_text(size=8)
            ) +\
            labs(title=title, x="Dim1", y="Dim2")

    else:
        raise ValueError("invalid dimension")


def gg_point_ident(
    df_in,
    colors=None,
    categories=None,
    title='Z-embeded',
    figure_size=(4,4)
):
    #print(f'title: {title}')
    assert df_in.shape[1] == 3, "requires three columns: x, y, ident"
    
    df = df_in.copy()
    df.columns = ["x", "y", "ident"]

    # weired category color
    if categories is not None:
        df['ident'] = pd.Categorical(df['ident'], categories=categories, ordered=True)

    #plot_margin=dict(left=1, right=1, top=1, bottom=1) # issues with writing to file
    #  axis.text = element_text(colour = "red", size = rel(1.5)) # relative size of text
    g = ggplot() +\
        geom_point(data=df, mapping=aes(x="x",y="y", color="ident"),
                                #size=1, position="jitter", alpha=0.7) +\
                                size=0.01, position="jitter", alpha=0.6) +\
        theme_bw() +\
        theme(
                 figure_size=figure_size,
                 panel_grid=element_blank(),
                 #legend_position="bottom", 
                 legend_background=element_blank(),
                 legend_title=element_blank(),
                 legend_key = element_blank(),
                 text = element_text(size=8)
        ) +\
        labs(title=title, x="Dim1", y="Dim2") +\
        guides(colour = guide_legend(override_aes={"size": 3}))

    
    if colors is not None:
        g = g + scale_color_manual(values = colors)

    return g


def heatmap_sample_idx(
    mtx: Union[pd.DataFrame, np.ndarray],
    idx_row, 
    idx_col,
    xlab="Feature", 
    ylab="Sample", 
    title="", 
    figure_size=(5,4)
):
    df = mtx
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        
    df_sampled = df.iloc[idx_row, idx_col]

    df_sampled.columns = list(range(df_sampled.shape[1]))
    df_sampled.index = list(range(df_sampled.shape[0]))

    return heatmap_from_mtx(df_sampled, xlab=xlab, ylab=ylab, title=title, figure_size=figure_size)

def heatmap_from_mtx(
    mtx: Union[pd.DataFrame, np.ndarray], 
    xlab="", 
    ylab="", 
    title="", 
    figure_size=(5,4),
    legend_position='right'
):
    df = mtx
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.T

    df_scaled = pd.DataFrame(color_scale_standard(df.values))
    #df_scaled = pd.DataFrame(color_scale_minmax(df.values))
    df_scaled.index = df.index
    df_scaled.columns = df.columns

    df = df_scaled

    df['x'] = df.index
    melted = df.melt(id_vars=["x"], var_name=["y"], value_name="score")
    melted

    #scale_color_manual(values=["white", "red"]) +\
    return ggplot(melted) +\
            geom_tile(aes(x="x", y="y", fill="score")) +\
            scale_fill_gradient2(low = "navy", mid = "white", high = "red", midpoint = 0) +\
            theme(
                legend_position=legend_position,
                figure_size=figure_size,
                panel_background=element_blank(),
                panel_spacing=element_blank(),
                panel_grid=element_blank(),
                axis_ticks=element_blank(),
                axis_text_x=element_blank(),
                axis_text_y=element_blank()) +\
            labs(title=title, x=xlab, y=ylab)


def gg_point_scatter(lst_1st, lst_2nd,
    figure_size = (3,3),
    color = 'red',
    xlab = 'x',
    ylab = 'y',
):
    
    df_scatter = pd.DataFrame(zip(lst_1st, lst_2nd), columns=['x', 'y'])
    
    return ggplot() +\
        	geom_point(data=df_scatter, mapping=aes(x="x",y="y"), color=color,
                                        size=1, position="jitter", alpha=0.7) +\
        	geom_abline(alpha=0.5, color='gray', linetype='dashed') +\
        	xlim(-1, 1) +\
        	ylim(-1, 1) +\
        	theme_bw() +\
        	theme(
            	    panel_grid=element_blank(),
            	    figure_size=figure_size
        	) +\
        	labs(x=xlab, y=ylab)


def grid_display(
    lst_imgs,
    ncols = 1,
    lst_titles = [],
    figratio = 1.0,
    filepath_out = 'merged.png',
    dpi=72
):
    nall = len(lst_imgs)
    nrows = int(np.ceil(nall/ncols))
    #print(nall, nrows, ncols); assert False

    ''' define figure size '''
    h, w = plt.imshow(plt.imread(lst_imgs[0])).get_size()
    plt.close()

    width = (w * ncols / dpi) * figratio
    height = (h * nrows / dpi) * figratio

    figsize = (width, height) 
    
    ''' if only single file '''
    if nrows == 1 and ncols == 1:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plt.imshow(plt.imread(lst_imgs[0]))
        if len(lst_titles) > 0:
            ax.set_title(lst_titles[0])
        ax.axis('off')
        fig.savefig(filepath_out)
        return

    ''' create figure '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    
    idx_img = 0
    for irow in range(nrows):
        for icol in range(ncols):

            ''' handle dimension in case of single row '''
            if nrows == 1 and ncols > 1:
                axe = axes[icol]
            elif ncols == 1 and nrows > 1:
                axe = axes[irow]
            else:
                axe = axes[irow, icol]

            ''' plot figure '''
            try:
                img = plt.imread(lst_imgs[idx_img])
                axe.imshow(img)
                axe.set_title(lst_titles[idx_img])
            except:
                axe.plot()
                #lst_titles.append('.')

            #if len(lst_titles) > 0:
                #axe.set_title(lst_titles[idx_img])

            axe.axis('off')
            idx_img += 1
            #plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.0)
            plt.tight_layout()

    ''' save figure '''
    fig.savefig(filepath_out)


def plot_violin():
    figure_size=(4,2)
    #colors = colors
    #geom_point(size = 1.2, col = alpha('grey', 0.2)) +
    #geom_boxplot(width=0.1) + theme_minimal()

    g = ggplot(df, aes(x='ident', y='score', fill='ident')) +\
        geom_violin(width=2) +\
        geom_boxplot(width=0.1) +\
        theme_bw() +\
        theme(
            panel_grid=element_blank(),
            figure_size=figure_size
        ) 
    g = g + scale_fill_manual(values = colors)
    g

    #    geom_jitter(position=position_jitter(0.1))
