
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math
import os
# -------------------data reading------------------------------------------------------
df = pd.read_excel("final projesi/dataset.xlsx")
label = pd.read_excel("final projesi/index.xlsx")
df.head()
df.tail()
# --------------------Normalization(max-min)-------------------------------------------


def max_min_normal(dataset, col):
    max_col = dataset[col].max()
    min_col = dataset[col].min()
    if (max_col-min_col) != 0: # To avoid division by zero
        dataset[col] = (dataset[col] - min_col) / (max_col - min_col)
    return dataset[col]


for col in df.columns:
    df[col] = max_min_normal(df, col)

# ---------------------------------SOM MODEL------------------------------------------


def euclidean_distance(col1, col2):
    distance = 0
    for i in range(len(col1)):
        distance += (col1[i] - col2[i]) ** 2
    return distance


def weight_matris_constructor(cluster_count, col_count):
    weight_matris = np.random.random((cluster_count, col_count))
    weight_matris = np.round(weight_matris, decimals=5)
    return weight_matris


def accuracy(cluster_pre, cluster_y):
    label_unique = cluster_y.unique()
    acc_list = {}
    for i in label_unique:
        acc = sum((cluster_pre == i) & (cluster_y == i)) / sum((cluster_y == i))
        acc_list[i] = acc
    return acc_list


def weight_update(weight_vect, row_vect, learning_rate, sigma):
    distance = euclidean_distance(row_vect, weight_vect)
    new_weight_vect = np.add(weight_vect, ((learning_rate * (np.exp(-(distance ** 2) / (2 * (sigma ** 2))))) * np.subtract(row_vect, weight_vect)))
    return new_weight_vect


def SOM(cluster_names, database, result, cluster_count, epoch, file=True, learning_rate=0.4, sigma=0.7, decey=0.9):
    w = weight_matris_constructor(cluster_count, len(database.columns))
    for i in range(0, epoch):
        result.clear()
        for row in range(0, len(database)):
            winner = {}
            for j in range(0, 3):
                winner.update({cluster_names[j]: (euclidean_distance(database.loc[row], w[j]))})
            result.append(min(winner, key=winner.get))
            index = list(winner).index(min(winner, key=winner.get))
            learning_rate = learning_rate * decey
            sigma = sigma * decey
            w[index] = weight_update(w[index], database.loc[row].values, learning_rate, sigma)
    if file:
        res = pd.DataFrame(result)
        res.to_csv("final projesi\\kume_sonucu.csv", header=True)
    return w

# --------------------------func çağırma--------------------------------
res = []
SOM([0, 3, 8, 9], df, res, 4, 75, True)
# ---------------------------sonuç değerleri okuma-----------------------------
kume_sonuc = pd.read_csv("final projesi/kume_sonucu.csv")
# ---------------------------bazı düzeltmeler-----------------------------
kume_sonuc = kume_sonuc.drop("Unnamed: 0", axis=1)
kume_sonuc.columns = ["label"]
# ----------------------------accuracy------------------------------------

acc = accuracy(kume_sonuc["label"], label["label"])
print(acc)



