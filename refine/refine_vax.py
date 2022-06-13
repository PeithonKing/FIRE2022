import pandas as pd
import json

path = "../raw/"
file = path+"vax_train.csv"

dataX = pd.read_csv(file).T.to_dict()

data = {}
for x in dataX.values():
    ys = x["label"]
    if ys == "AntiVax": y = 0
    elif ys == "Neutral": y = 1
    elif ys == "ProVax": y = 2
    else:
        print(f"ys = {ys}\ndisaster")
        break
    data[str(x['id'])] = {"tweet":x['tweet'],
                     "y": y}

with open("../vax/data/data.json", "w") as f:
    json.dump(data, f)