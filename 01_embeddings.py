import glob

import networkx as nx
import pandas as pd

# TODO fix imports
from openne import gf, graph, hope, lap, node2vec, sdne


def prepare_dataset():
    df = pd.read_csv('dataset.csv', delimiter='\t')
    df = df.drop("Disease Name", axis=1)
    # used later to distinguish diseases column from genes one
    df["# Disease ID"] = '#' + df["# Disease ID"]
    df.to_csv("dataset.txt", header=None, index=None, sep=' ')


def get_adj_matrix():
    df = pd.read_csv('dataset.txt', delimiter=' ', header=None)
    g = nx.from_pandas_edgelist(df, source=1, target=0)
    return nx.adjacency_matrix(g), list(g.nodes())


def get_hope(g):
    embeddings = hope.HOPE(graph=g, d=128)
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
    g.read_edgelist("dataset.txt")

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
    #gene_dis_matrix = np.matrix(
    #    np.loadtxt("./DG-Miner_miner-disease-gene.tsv",
    #               delimiter=',', dtype=int)
    #)

    if "./dataset.txt" not in glob.glob("./*.txt"):
        prepare_dataset()

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
