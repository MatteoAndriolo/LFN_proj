import glob

import networkx as nx
import pandas as pd
import logging as lg

# TODO fix imports
from openne import gf, graph, hope, lap, node2vec, sdne


def prepare_dataset(namedataset):
    df = pd.read_csv(namedataset, delimiter='\t')
    #df = df.drop("# Disease(MESH)", axis=1)
    # used later to distinguish diseases column from genes one
    #df["# Gene ID"] = '#' + df["# Gene"]
    df.to_csv("dataset.txt", header=None, index=None, sep=' ')
    lg.info("prepare_dataset: dataset completed")


def get_adj_matrix():
    df = pd.read_csv('dataset.txt', delimiter=' ', header=None)
    lg.info("get_adj_matrix: dataset read")
    g = nx.from_pandas_edgelist(df, source=1, target=0)
    lg.info("get_adj_matrix: networkx graph generated")
    return nx.adjacency_matrix(g), list(g.nodes())


def get_hope(g):
    embeddings = hope.HOPE(graph=g, d=128)
    lg.info("get_hope: embeddings generated")
    return embeddings


def get_sdne(g):
    embeddings = sdne.SDNE(g, [1000, 128])
    return embeddings


def get_n2v(g):
    embeddings = node2vec.Node2vec(
        graph=g, path_length=80, num_paths=10, dim=10)
    return embeddings


def get_dw(g):
    embeddings = node2vec.Node2vec(
        graph=g, path_length=80, num_paths=10, dim=10, dw=True)
    return embeddings


def get_gf(g):
    embeddings = gf.GraphFactorization(g)
    return embeddings


def get_lap(g):
    embeddings = lap.LaplacianEigenmaps(g)
    return embeddings


def get_embeddings(t):
    """
    Parameters:
    t : str
      type accepted [hope, sdne, n2v, dw, gf, lap]
    """
    g = graph.Graph()
    lg.info("get_embeddings: read_edgelist starting")
    g.read_edgelist("dataset.txt")
    lg.info("get_embeddings: edgelist read")

    
    lg.info("get_embeddings: start generation of embeddings")
    if t == "hope":
        embeddings = get_hope(g)
    elif t == "sdne":
        embeddings = get_sdne(g)
    elif t == "n2v":
        embeddings = get_n2v(g)
    elif t == "dw":
        embeddings = get_dw(g)
    elif t == "gf":
        embeddings = get_gf(g)
    elif t == "lap":
        embeddings = get_lap(g)
    else:
        return [], []

    dis_emb, gen_emb = [], []  # split embeddings in diseases and genes ones
    for k, v in embeddings.vectors.items():
        if k[0] == '#':
            dis_emb.append(v)
        else:
            gen_emb.append(v)
    return dis_emb, gen_emb



################################################################################
if __name__ == "__main__":
    lg.basicConfig(level=lg.INFO)
    #gene_dis_matrix = np.matrix(
    #    np.loadtxt("./DG-Miner_miner-disease-gene.tsv",
    #               delimiter=',', dtype=int)
    #)

    namedataset="DG-Miner_miner-disease-gene.tsv"
    if namedataset not in glob.glob("./*"):
        prepare_dataset(namedataset)

    # needed to find labels of (disease,gene) pairs
    adj_matrix, nodes = get_adj_matrix()

    embeddings = get_embeddings("hope")


    # result=np.zeros((1, 7), float)
    # average_result=np.zeros((1, 7), float)
    # circle_time=1  # seed

    # for i in range(circle_time):
    #     result += cross_validation_experiment(gene_dis_matrix, i, 1)

    # average_result=result/circle_time
    # print(average_result)
