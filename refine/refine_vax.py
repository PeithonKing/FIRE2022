import pandas as pd

path = "../raw/"
file = path+"vax_train.csv"

dataX = pd.read_csv(file).T.to_dict()

data = {}
for i, x in enumerate(dataX.values()):
    y = {"tweet": x['tweet'], "AntiVax": 0, "Neutral": 0, "ProVax": 0}
    y[x["label"]] = 1
    data[i] = y

data = pd.DataFrame(data).T
# print(data.head())
data.to_csv("../vax/data/data.csv", index=False)