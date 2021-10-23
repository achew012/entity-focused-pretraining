from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math, ipdb

# Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='ner-pretraining', task_name='BIO-Loss', output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

# config = json.load(open('config.json'))

config={
    "lr": 3e-4,
    "num_epochs":50,
    "train_batch_size":4,
    "eval_batch_size":1,
    "max_length": 2048, # be mindful underlength will cause device cuda side error
    "use_entities_as_spans": True, # Toggle between entity pretraining or span pretraining
    "mlm_task": False,
    "sbo_task": False,
    "bio_task": True,
}
args = argparse.Namespace(**config)
task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.execute_remotely(queue_name="128RAMv100", exit_process=True)
task.connect(args)

dataset = ds.get(dataset_name="processed-DWIE", dataset_project="datasets/DWIE", dataset_tags=["1st-mention"], only_published=True)
dataset_folder = dataset.get_local_copy()
dwie = json.load(open(os.path.join(dataset_folder, "data", "new_dwie.json")))["dataset"]

import os, random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerLMHead, _compute_global_attention_mask
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import classification_report
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

class DWIE_Data(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.consolidated_dataset = []
        self.max_spans = 20 # 15 -> 95 , 25 -> 88, 20 -> 92 docs
        self.max_span_len = 15

        for idx, doc in enumerate(self.dataset):
            context = ' '.join(doc["sentences"])
            tokens = self.tokenizer.tokenize(context)
            context_len = len(tokens)
            spans = enumerate_spans(tokens, min_span_width=1, max_span_width=self.max_span_len)

            self.encodings = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
            
            if len(doc["entities"])>0:
                doc_entity_mask = torch.zeros(self.tokenizer.model_max_length)
                # bio_labels = torch.full_like(self.encodings["input_ids"], -100)[0]
                # bio_labels[:context_len] = self.class2id["O"]
                
                #entities_samples = random.choices(doc["entities"], k=self.max_spans)
                #entities_samples = random.sample(doc["entities"], k=self.max_spans)
                entity_spans = []
                for idx, entity in enumerate(doc["entities"]):
                    entity_start = entity["start"] if entity["start"] < self.tokenizer.model_max_length else 0
                    entity_end = entity["end"] if entity["end"]<self.tokenizer.model_max_length else self.tokenizer.model_max_length-1

                    pair = (entity_start-1 if entity_start>0 else 0, entity_end)
                    if pair in spans:
                        # get mask of entities
                        doc_entity_mask[entity_start:entity_end] = idx 
                        entity_span = spans.pop(spans.index(pair))
                        # entity_span = (entity_span[0]+1,entity_span[1]+1) #offset the CLS
                        entity_spans.append(torch.tensor(entity_span))
                    
                entities_samples = random.choices(entity_spans, k=self.max_spans)
                negative_samples = [torch.tensor(sample) for sample in random.sample(spans, k=self.max_spans)]

                labels = torch.tensor([1.0]*len(entities_samples)+[0.0]*len(negative_samples))
                samples = entities_samples + negative_samples

                self.consolidated_dataset.append({
                    "input_ids": self.encodings["input_ids"],
                    "attention_mask": self.encodings["attention_mask"],
                    "entity_span": torch.stack(samples),
                    "entity_mask": doc_entity_mask,
                    "labels": labels
                })           

    def __len__(self):
        return len(self.consolidated_dataset)

    def __getitem__(self, idx):
        item = self.consolidated_dataset[idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).squeeze(1)
        entity_span = torch.stack([ex['entity_span'] for ex in batch])
        entity_mask = torch.stack([ex['entity_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        
        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "entity_span": entity_span,
                "entity_mask": entity_mask,
                "labels": labels
        }

class NERLongformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        self.config.gradient_checkpointing = True

        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer.model_max_length = self.args.max_length
        # self.longformer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)

        self.longformer = LongformerModel(self.config)
        #self.lm_head = LongformerLMHead(self.config) 

        # train_data = DWIE_Data(dwie["train"], self.tokenizer, self.class2id, self.args)
        # y_train = torch.stack([doc["bio_labels"] for doc in train_data.consolidated_dataset]).view(-1).cpu().numpy()
        # y_train = y_train[y_train != -100]
        # self.class_weights=torch.cuda.FloatTensor(compute_class_weight("balanced", np.unique(y_train), y_train))
        # #self.class_weights[0] = self.class_weights[0]/2
        # self.class_weights = torch.cuda.FloatTensor([0.20, 1, 1.2, 1, 1.2])
        # print("weights: {}".format(self.class_weights))

        # self.dropout = nn.Dropout(self.config.hidden_dropout_prob)       
        # self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Linear(self.config.hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def val_dataloader(self):
        val = dwie["test"]
        val_data = DWIE_Data(val, self.tokenizer, self.args)
        val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
        return val_dataloader

    def train_dataloader(self):
        train = dwie["train"]
        train_data = DWIE_Data(train, self.tokenizer, self.args)
        train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, collate_fn = train_data.collate_fn)
        return train_dataloader

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        global_attention_mask[:, :1] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask


    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions


        entity_span = batch.pop("entity_span", None)
        entity_mask = batch.pop("entity_mask", None)
        labels = batch.pop("labels", None)

        # input_ids, attention_mask, entity_span, entity_mask = batch["input_ids"], batch["attention_mask"], batch["entity_span"], batch["entity_mask"]

        # if "labels" in batch.keys():
        #     labels = batch["labels"]

        outputs = self.longformer(
            **batch, 
            global_attention_mask=self._set_global_attention_mask(batch["input_ids"]), output_hidden_states=True
            )

        sequence_output = outputs[0] 

        logits = torch.cuda.FloatTensor([])
        for seq, spans in zip(sequence_output, entity_span):
            cls_embedding = seq[1]
            seq_embedding = seq[1:]
            mean_tokens_embeddings = torch.cuda.FloatTensor([])
            for span in spans:
                span_embeds = seq_embedding[torch.arange(span[0], span[1]+1)]
                combined_embeds = torch.cat([cls_embedding, span_embeds.mean(dim=0)], dim=0)
                span_logits = self.sigmoid(self.classifier(combined_embeds))
                mean_tokens_embeddings = torch.cat([mean_tokens_embeddings, span_logits], dim=0)
            logits = torch.cat([logits, mean_tokens_embeddings], dim=0)

        total_loss=None
        if labels!=None:
            loss_fct = nn.BCELoss()
            total_loss = loss_fct(logits, labels)

        return (total_loss, logits)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _ = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
        }

        self.log("train_loss", logs["train_loss"])

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        preds = (logits>0.5)
        # print(hidden_states)
        return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_preds = torch.stack([x["preds"] for x in outputs]).view(-1).cpu().detach().tolist()
        val_labels = torch.stack([x["labels"] for x in outputs]).view(-1).cpu().detach().tolist()

        print(val_preds)

        logs = {
            "val_loss": val_loss_mean,
        }

        self.log("val_loss", logs["val_loss"])

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        # for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
        #     if idx%2==0:
        #         parameters.requires_grad=False
        #     else:
        #         parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.longformer.parameters(), lr=self.args.lr)
        return optimizer

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_entity_lm", 
    monitor="val_loss", 
    mode="min", 
    save_top_k=1, 
    save_weights_only=True,
    period=3,
)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=8, verbose=False, mode="min")
NERLongformer = NERLongformer(args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(NERLongformer)
