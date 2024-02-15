import logging
from datetime import datetime

from pathlib import Path
from utils import c, init_logs, get_logger
init_logs(True, logging.DEBUG)

c("Importing ML frameworks")
import torch
import numpy as np
from tqdm import tqdm

c("Importing model")
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification

c("Importing data functions")
from datasets import load_dataset
import evaluate

c("Importing ONNX utilities")
from transformers.onnx import FeaturesManager, export

from huggingface_hub import login

login("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")

DATASET = "nlpaueb/finer-139"
MODEL = "distilbert-base-uncased"
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-05
IGNORE_LABEL = -100

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class Finer139DataSet:    
    def __init__(self, tokenizer: AutoTokenizer, mode: str = "train", only_predict_numbers: bool=False):
        self.tokenizer = tokenizer
        self.only_predict_numbers = only_predict_numbers
        
        c(f"Loading data: {mode}")
        self._data = load_dataset(DATASET)[mode]
        
        c("Realigning tokenized input and ner tags")
        self.tokenized_input = self._data.map(self.tokenize_and_align_labels, batched=True)

    def _is_not_a_number(self, token: str):
        try:
            float(token)
            return False
        except ValueError:
            return True

    def _skip_token(self, token: str):
        if not self.only_predict_numbers:
            return False
        return self._is_not_a_number(token)

    def tokenize_and_align_labels(self, batch):
        tokenized_inputs = self.tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
        
        labels = []
        for i, label in enumerate(batch["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(IGNORE_LABEL)
                elif word_idx != previous_word_idx and not self._skip_token(batch["tokens"][i][word_idx]):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(IGNORE_LABEL)
                previous_word_idx = word_idx
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_ids2labels(self):
        return {k:v for k, v in enumerate(self._data.features["ner_tags"].feature.names)}
    
    def get_labels2ids(self):
        return {v:k for k, v in enumerate(self._data.features["ner_tags"].feature.names)}

    def __getitem__(self, idx):
        return self.tokenized_input[idx]
    
    def __len__(self):
        return len(self.tokenized_input)

c("Loading seqeval")
seqeval = evaluate.load("seqeval")

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)

def get_label_list():
    data = load_dataset(DATASET)["test"]
    return data.features["ner_tags"].feature.names

LABEL_LIST = get_label_list()

def compute_metrics(predictions_and_targets):
    predictions, labels = predictions_and_targets

    true_predictions, true_labels = [], []
    for i in tqdm(range(predictions.shape[0])):
        seq_preds = predictions[i]
        seq_labels = labels[i]
        assert len(seq_preds) == len(seq_labels)
        true_pred, true_label = [], []
        contains_number = False
        for j in range(len(seq_preds)):
            if seq_labels[j] == IGNORE_LABEL: continue
            true_pred.append(LABEL_LIST[predictions[i, j]])
            true_label.append(LABEL_LIST[labels[i, j]])
            if labels[i, j] != 0:
                contains_number = True
        if contains_number:
            true_predictions.append(true_pred)
            true_labels.append(true_label)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1" : results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

def preprocess_logits_for_metrics(logits, labels):
        return torch.argmax(logits, dim=-1)

def execute():
    log = get_logger()
    
    strat = "epoch"
    c("Setting Training arguments")
    train_args = TrainingArguments(
        output_dir="models/DataSnipper_FinerDistilBert_FullSequence",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TEST_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy=strat,
        save_strategy=strat,
        eval_accumulation_steps=2000,
        load_best_model_at_end=True
    )

    c("Creating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    c("Creating train and test set")
    trainset = Finer139DataSet(tokenizer, "train", only_predict_numbers=False)
    testset = Finer139DataSet(tokenizer, "test", only_predict_numbers=True)

    c("Creating data collator")
    data_collator = DataCollatorForTokenClassification(tokenizer=trainset.tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL,
        num_labels=len(trainset.get_ids2labels().keys()),
        id2label=trainset.get_ids2labels(),
        label2id=trainset.get_labels2ids()
    )

    c("Creating Trainer")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

    c("Training model")
    trainer.train()

    save_path = f"models/local/{datetime.now().strftime('%Y%m%d_%H%M%S')}_finer_distilbert"
    torch.save(model.state_dict(), save_path + ".pth")

    trainer.push_to_hub("Finetuning done")

    log.info("Execution successful!")
    return 0

if __name__ == "__main__":
    execute()