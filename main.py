from transformers import BertTokenizer
from custom_dataset import CustomDataset
from models.train_bert import train_bert

import pandas as pd

if __name__ == "__main__":
    print("----- Loading Data -----")
    df = pd.read_csv("data/CombinedDataset.csv")  # or whatever your cleaned CSV is
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = CustomDataset(texts, labels, tokenizer, max_length=512)

    print("----- Training BERT -----")
    train_bert(dataset)
