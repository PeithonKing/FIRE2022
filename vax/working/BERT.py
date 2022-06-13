import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from tqdm import tqdm

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
data = pd.read_csv("../data/data.csv")

def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

batch_size = 200
div = np.append(np.arange(0, 4392, batch_size), 4392)
for i in tqdm(range(len(div)-1)):
    start = div[i]
    end = div[i+1]
    diri = {'AntiVax': np.array(data["AntiVax"][start:end]),
            'Neutral': np.array(data["Neutral"][start:end]),
            'ProVax': np.array(data["ProVax"][start:end])}
    crouch = get_sentence_embeding(data["tweet"][start:end])
    for j in range(768): diri[f"X{j+1}"] = np.array(crouch)[:, j]
    pd.DataFrame(diri).to_csv(f"{batch_size}/data_{start}_{end-1}.csv", index=False)
    del crouch