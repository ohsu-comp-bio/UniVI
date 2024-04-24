import pandas as pd
from scvt.plotting.plots import gg_point_ident

def cluster_louvain(
    mtx,
    n_neighbors=10,
    
):
    from sklearn.neighbors import NearestNeighbors
    import networkx as nx
    import community as community_louvain
    from scanpy import neighbors
    #from scanpy import _utils
    
    n_obs = mtx.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(mtx)
    distances, indices = nbrs.kneighbors(mtx)
    distances, connectivities = neighbors._compute_connectivities_umap(indices, distances, n_obs=n_obs, n_neighbors=n_neighbors)
    #g = _utils.get_igraph_from_adjacency(connectivities)
    g = nx.Graph(connectivities)
    partition = community_louvain.best_partition(g)
    return partition


def cluster_louvain_plot(
    mtx,
    n_neighbors=10,    
):    
    partition = cluster_louvain(mtx, n_neighbors=n_neighbors)
    #set(list(partition.values()))
    df_ident = pd.DataFrame(mtx)
    #df_ident['ident'] = list(partition.values())
    df_ident['ident'] = pd.Categorical(list(partition.values()))
    df_ident.columns = ["Dim1", "Dim2", "Ident"]
    g = gg_point_ident(df_ident)
    print(g)
    return partition
