import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics as ev
import os

def g(x):
    return 1 / (1 + np.exp(-x))
def h(x, theta):
    return g(x@theta)
def J(theta, X, Y):
    return np.mean((Y - 1) * np.log(1 - h(X, theta)) - Y * np.log(h(X, theta)))

imps = ["AntiVax", "Neutral", "ProVax"]
iterations = int(1.2*10**6)

for imp in imps:
    print(f"Starting with {imp}.")
    alpha = 0.0125
    # Load Train dataset
    X = pd.read_csv("../data/BERT/X_train.csv").to_numpy()
    Y = pd.read_csv("../data/BERT/Y_train.csv")[imp].to_numpy().reshape(X.shape[0],1)
    if os.path.exists(f"results/theta_{imp}.npy"):
        print(f"Loading theta_{imp}")
        theta = np.load(f"results/theta_{imp}.npy")
    else:
        print("Could not find theta file, so starting with random theta.")
        theta = np.random.rand(X.shape[1], 1)
    
    # Load Test Data
    X_test = pd.read_csv("../data/BERT/X_test.csv").to_numpy()
    Y_test = pd.read_csv("../data/BERT/Y_test.csv")[imp].to_numpy().reshape(X_test.shape[0],1)
    
    # # Do Gradient Descent and get optimum theta
    Js = {}
    for i in tqdm(range(iterations)):
        try:
            newJ = J(theta, X, Y)
            Js[i] = newJ
            slope = (X.T @ (h(X, theta) - Y))
            theta -= alpha * slope / len(X)
        except:
            break
        if not (i%10**3) and i:
            # print(f"\ncurrent J = {newJ}\n")
            with open("updates.txt", "w") as f: f.write(str(newJ))
    
    
    roc = ev.roc_curve(Y_test, h(X_test, theta), pos_label=1)
    plt.plot(roc[0], roc[1])
    plt.title(f"ROC Curve for {imp}")
    plt.savefig(f"results/{imp}.png")
    
    np.save(f"results/theta_{imp}.npy", theta)
    
    print(f"\n{imp} best J = {min(Js.values())}\n")
