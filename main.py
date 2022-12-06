import gc
import random
import glob
import random as rd

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import KFold

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


# TODO fix this function
def Calculate_metrics(predict_y_proba, test_feature_matrix, test_label_vector):

    clf = RandomForestClassifier(
        random_state=1, n_estimators=200, oob_score=True, n_jobs=-1)
    clf.fit(train_feature_matrix, train_label_vector)
    predict_y_proba = clf.predict_proba(test_feature_matrix)[:, 1]
    predict_y = clf.predict(test_feature_matrix)
    AUPR = average_precision_score(test_label_vector, predict_y_proba)
    AUC = roc_auc_score(test_label_vector, predict_y_proba)
    MCC = matthews_corrcoef(test_label_vector, predict_y)
    ACC = accuracy_score(test_label_vector, predict_y, normalize=True)
    F1 = f1_score(test_label_vector, predict_y, average='binary')
    REC = recall_score(test_label_vector, predict_y, average='binary')
    PRE = precision_score(test_label_vector, predict_y, average='binary')
    metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
    print(metric)

    del train_feature_matrix
    del test_feature_matrix
    del train_label_vector
    del test_label_vector
    gc.collect()
    return metric


def cross_validation_experiment(gene_dis_matrix, seed, ratio=1):
    none_zero_position = np.where(gene_dis_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(gene_dis_matrix == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]
    random.seed(seed)
    zero_random_index = random.sample(
        range(len(zero_row_index)), ratio * len(none_zero_row_index))
    zero_row_index = zero_row_index[zero_random_index]
    zero_col_index = zero_col_index[zero_random_index]

    row_index = np.append(none_zero_row_index, zero_row_index)
    col_index = np.append(none_zero_col_index, zero_col_index)

    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    metric = np.zeros((1, 7), float)
    print("seed=%d, evaluating gene-disease...." % (seed))
    k_count = 0

    for train, test in kf.split(row_index):

        train_gene_dis_matrix = np.copy(gene_dis_matrix)

        test_row = row_index[test]
        test_col = col_index[test]
        train_row = row_index[train]
        train_col = col_index[train]

        train_gene_dis_matrix[test_row, test_col] = 0

        # TODO fix t to specify for type embedding
        gene_disease_emb = get_embeddings(t)
        gene_len = gene_dis_matrix.shape[0]
        gene_emb_matrix = np.array(gene_disease_emb[0:gene_len, 1:])

        train_feature_matrix = []
        train_label_vector = []
# TODO fix lines after this
        for num in range(len(train_row)):
            feature_vector = np.append(
                gene_emb_matrix[train_row[num], :], dis_emb_matrix[train_col[num], :])
            train_feature_matrix.append(feature_vector)
            train_label_vector.append(
                gene_dis_matrix[train_row[num], train_col[num]])

        test_feature_matrix = []
        test_label_vector = []

        for num in range(len(test_row)):
            feature_vector = np.append(
                gene_emb_matrix[test_row[num], :], dis_emb_matrix[test_col[num], :])
            test_feature_matrix.append(feature_vector)
            test_label_vector.append(
                gene_dis_matrix[test_row[num], test_col[num]])

        train_feature_matrix = np.array(train_feature_matrix)
        train_label_vector = np.array(train_label_vector)
        test_feature_matrix = np.array(test_feature_matrix)
        test_label_vector = np.array(test_label_vector)

        clf = RandomForestClassifier(
            random_state=1, n_estimators=200, oob_score=True, n_jobs=-1)
        clf.fit(train_feature_matrix, train_label_vector)
        predict_y_proba = clf.predict_proba(test_feature_matrix)[:, 1]
        predict_y_proba += metric

        metric += Calculate_metrics(predict_y_proba,
                                    test_feature_matrix, test_label_vector)
        # k_count+=1

    # print(metric / kf.n_splits)

    metric_avg = np.zeros((1, 7), float)
    metric_avg[0, :] += metric / kf.n_splits
    metric = np.array(metric_avg)
    name = 'seed=' + str(seed) + '.csv'
    np.savetxt(name, metric, delimiter=',')
    return metric


def get_sample(emb_dict, a, b):
    to_stack = (emb_dict.get(a), emb_dict.get(b)) if '#' in a  else (emb_dict.get(b), emb_dict.get(a))
    return np.hstack(to_stack)  # horiz. stacked embeddings -> dis_emb|gen_emb


def get_zero_indexes(adj_matrix, num_ones):
    zero_rows, zero_cols = np.where(adj_matrix == 0)
    # pick n=num_ones random indices to have len(pos_samples)==len(neg_samples) and so achieve a balanced training
    random_zero_indexes = rd.sample(range(len(zero_rows)), num_ones)
    return zero_rows[random_zero_indexes], zero_cols[random_zero_indexes]


def get_stuff(adj_matrix, embeddings, nodes):
    # creating a matrix of only positive samples
    adj_matrix = np.triu(adj_matrix) # adjmatrix is simmetric -> triu -> no double count
    one_rows, one_cols = np.where(adj_matrix == 1)
    pos_samples = np.array([get_sample(embeddings, nodes[i], nodes[j])
                           for i, j in zip(one_cols, one_rows)])

    # creating a matrix of only negative samples
    zero_rows, zero_cols = get_zero_indexes(adj_matrix, len(one_rows))
    neg_samples = np.array([get_sample(embeddings, nodes[i], nodes[j])
                           for i, j in zip(zero_rows, zero_cols)])

    # creating a matrix of the mixed and shuffled samples along with its labels matrix
    samples, labels = shuffle(np.vstack((pos_samples, neg_samples)), np.hstack(
        (np.ones(len(pos_samples)), np.zeros(len(neg_samples)))))

    return train_test_split(samples, labels, test_size=0.3)


def train_and_predict(models, adj_matrix, embeddings, nodes):
    samples_train, labels_train, samples_test, labels_test = get_stuff(
        adj_matrix, embeddings, nodes)
    print("Training..")
    fitted_models = [m.fit(samples_train, labels_train) for m in models]
    print("Predicting..")
    predictions = [np.rint(m.predict(samples_test)) for m in fitted_models]
    return [accuracy_score(labels_test, p) for p in predictions]


if __name__ == "__main__":
    gene_dis_matrix = np.matrix(
        np.loadtxt("./DG-Miner_miner-disease-gene.tsv",
                   delimiter=',', dtype=int)
    )

    if "./dataset.txt" not in glob.glob("./*.txt"):
        prepare_dataset()

    # needed to find labels of (disease,gene) pairs
    adj_matrix, nodes = get_adj_matrix()

    embeddings = get_embeddings("hope")

    models = [
        RandomForestClassifier(n_estimators=200, n_jobs=-1),
        AdaBoostClassifier(n_estimators=200)
    ]

    print(train_and_predict(models, adj_matrix, embeddings, nodes))

    # result=np.zeros((1, 7), float)
    # average_result=np.zeros((1, 7), float)
    # circle_time=1  # seed

    # for i in range(circle_time):
    #     result += cross_validation_experiment(gene_dis_matrix, i, 1)

    # average_result=result/circle_time
    # print(average_result)
