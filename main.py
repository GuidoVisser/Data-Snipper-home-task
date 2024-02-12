import logging
from utils import c, init_logs, get_logger
init_logs(True, logging.DEBUG)

c("Importing ML frameworks")
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

c("Importing utilities")
from tqdm import tqdm
from sklearn import metrics

c("Importing data classes")
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

c("Importing model")
from transformers import DistilBertTokenizer, DistilBertModel

c("Importing data functions")
from datasets import load_dataset

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class Finer139DataSet:
    def __init__(self, mode: str = "train", max_length=128):
        self._max_length = max_length
        self._data = load_dataset("nlpaueb/finer-139")[mode]
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
        self._nr_features = len(self._data.features["ner_tags"].feature.names)
        
    def __getitem__(self, idx):
        datapoint = self._data[idx]
        
        inputs = self.tokenizer.encode_plus(
            " ".join(datapoint["tokens"]),
            None,
            add_special_tokens=True,
            max_length=self._max_length,
            padding="max_length",
            return_token_type_ids=False
        )
        tags = datapoint["ner_tags"]

        pad_size = max(len(inputs["input_ids"]), self._max_length) - len(tags)

        return {
            "ids" : torch.Tensor(inputs["input_ids"], device=get_device()),
            "mask": torch.Tensor(inputs["attention_mask"], device=get_device()),
            "tags": F.one_hot(
                F.pad(
                    input=torch.Tensor(tags, device=get_device()), 
                    pad=(0, pad_size), 
                    mode="constant", 
                    value=0).to(dtype=torch.int64),
                num_classes=self._nr_features).to(dtype=torch.float32)
        }
        
    def __len__(self):
        return len(self._data)
    
class FinerDistilBert(nn.Module):
    def __init__(self, nr_features=279) -> None:
        super().__init__()
        self._bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self._out_layer = nn.Linear(768, nr_features)

    def __call__(self, inputs):
        bert_out = self._bert(
            input_ids=inputs["ids"].to(device=get_device(), dtype=torch.long),
            attention_mask=inputs["mask"].to(device=get_device(), dtype=torch.long)
        )
        return self._out_layer(bert_out.last_hidden_state)

    def save(self):
        return
    
    def load(self):
        return


def loss_fn(model_out, targets):
    return nn.CrossEntropyLoss()(model_out, targets)

def execute():
    log = get_logger()
    
    MAX_LEN = 2048
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 1
    EPOCHS = 1
    LEARNING_RATE = 1e-05    
    
    train_set = Finer139DataSet("train", max_length=MAX_LEN)
    valid_set = Finer139DataSet("validation", max_length=MAX_LEN)
    test_set  = Finer139DataSet("test", max_length=MAX_LEN)
    total_nr_tags = len(list(set(train_set._data.features['ner_tags'].feature.names) | \
                    set(valid_set._data.features['ner_tags'].feature.names) | \
                    set(test_set._data.features['ner_tags'].feature.names)))
    
    log.info(f"Number of tags available")
    log.info(f"  - Train:    {len(train_set._data.features['ner_tags'].feature.names)}")
    log.info(f"  - Validate: {len(valid_set._data.features['ner_tags'].feature.names)}")
    log.info(f"  - Test:     {len(test_set._data.features['ner_tags'].feature.names)}")
    log.info(f"  - Total:    {total_nr_tags}")
    
    testloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
    trainloader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    model = FinerDistilBert(nr_features=total_nr_tags)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    max_length = 0
    for batch in tqdm(trainloader):
        output = model(batch)
        
        optimizer.zero_grad()
        loss = loss_fn(output, batch["tags"])
        loss.backward()
        optimizer.step()

    log.info("Execution successful!")
    return 0

if __name__ == "__main__":
    execute()
    
    
# TODOs
# - ensure tokens and labels are alligned
# - Look into DataCollatorForTokenClassification
# - Look into AutoModelForTokenClassification
# - Convert to ONNX

# TODOs Bonus task
# - Run in Docker
# - Create an API 