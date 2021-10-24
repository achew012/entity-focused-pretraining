from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math, ipdb

# Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='ner-pretraining', task_name='EntitySpanPretraining', tags=["maxpool"], output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

# config = json.load(open('config.json'))

config={
    "lr": 5e-4,
    "num_epochs":5,
    "train_batch_size":12,
    "eval_batch_size":1,
    "max_length": 2048, # be mindful underlength will cause device cuda side error
    "max_span_len": 15,
    "max_spans": 25,
    "mlm_task": False,
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
from sklearn.metrics import classification_report, precision_recall_fscore_support

from typing import Callable, List, Set, Tuple, TypeVar, Optional

def enumerate_spans(
    sentence: List,
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List], bool] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.
    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.
    # Parameters
    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans

class DWIE_Data(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.consolidated_dataset = []
        self.max_spans = args.max_spans # 15 -> 95 , 25 -> 88, 20 -> 92 docs
        self.max_span_len = args.max_span_len

        for idx, doc in enumerate(self.dataset):
            context = ' '.join(doc["sentences"])
            tokens = self.tokenizer.tokenize(context)
            context_len = len(tokens)
            spans = enumerate_spans(tokens, min_span_width=1, max_span_width=self.max_span_len)

            self.encodings = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
            masked_input_ids = self.encodings["input_ids"][0].clone()
            
            if len(doc["entities"])>0:
                doc_entity_mask = torch.zeros(self.tokenizer.model_max_length)
                # bio_labels = torch.full_like(self.encodings["input_ids"], -100)[0]
                # bio_labels[:context_len] = self.class2id["O"]
                
                #entities_samples = random.choices(doc["entities"], k=self.max_spans)
                #entities_samples = random.sample(doc["entities"], k=self.max_spans)
                entity_spans = []
                entity_len_mask = []
                for idx, entity in enumerate(doc["entities"]):
                    entity_start = entity["start"] if entity["start"] < self.tokenizer.model_max_length else 0
                    entity_end = entity["end"] if entity["end"]<self.tokenizer.model_max_length else self.tokenizer.model_max_length-1

                    ent_masky = torch.zeros(self.max_span_len)

                    pair = (entity_start-1 if entity_start>0 else 0, entity_end)
                    if pair in spans:
                        masked_input_ids[pair[0]+1:pair[1]+1] = self.tokenizer.mask_token_id
                        # get mask of entities
                        doc_entity_mask[pair[0]:pair[1]] = idx+1
                        ent_masky[:(pair[1]-pair[0])] = torch.arange(pair[0], pair[1])
                        entity_span = spans.pop(spans.index(pair))
                        # entity_span = (entity_span[0]+1,entity_span[1]+1) #offset the CLS
                        entity_spans.append(torch.tensor(entity_span))
                        entity_len_mask.append(ent_masky)

                #entities_samples = random.choices(entity_spans, k=self.max_spans)
                selected_entities = [random.randint(0, len(entity_spans)-1) for i in range(self.max_spans)]
                entities_samples = [entity_spans[i] for i in selected_entities]
                entity_len_mask = [entity_len_mask[i] for i in selected_entities]
                negative_samples = [torch.tensor(sample) for sample in random.sample(spans, k=self.max_spans)]

                noise_len_mask = []
                for negative_sample in negative_samples:
                    entity_start = negative_sample[0]
                    entity_end = negative_sample[1]

                    noise_masky = torch.zeros(self.max_span_len)
                    noise_masky[:(entity_end-entity_start)] = torch.arange(entity_start, entity_end)                        

                    doc_entity_mask[entity_start:entity_end] = -200
                    noise_len_mask.append(noise_masky)

                labels = torch.tensor([1.0]*len(entities_samples)+[0.0]*len(negative_samples))
                samples = entities_samples + negative_samples
                samples_mask = entity_len_mask + noise_len_mask

                self.consolidated_dataset.append({
                    "masked_input_ids": masked_input_ids,
                    "input_ids": self.encodings["input_ids"],
                    "attention_mask": self.encodings["attention_mask"],
                    "entity_span": torch.stack(samples),
                    "entity_mask": doc_entity_mask,
                    "span_len_mask": torch.stack(samples_mask),
                    "labels": labels
                })           

    def __len__(self):
        return len(self.consolidated_dataset)

    def __getitem__(self, idx):
        item = self.consolidated_dataset[idx]
        return item

    def collate_fn(self, batch):
        masked_input_ids = torch.stack([ex['masked_input_ids'] for ex in batch]).squeeze(1)
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).squeeze(1)
        entity_span = torch.stack([ex['entity_span'] for ex in batch])
        entity_mask = torch.stack([ex['entity_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch])
        span_len_mask = torch.stack([ex["span_len_mask"] for ex in batch])
        
        return {
                "masked_input_ids": masked_input_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "entity_span": entity_span,
                "entity_mask": entity_mask,
                "labels": labels,
                "span_len_mask": span_len_mask
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
        self.lm_head = LongformerLMHead(self.config) 

        # train_data = DWIE_Data(dwie["train"], self.tokenizer, self.class2id, self.args)
        # y_train = torch.stack([doc["bio_labels"] for doc in train_data.consolidated_dataset]).view(-1).cpu().numpy()
        # y_train = y_train[y_train != -100]
        # self.class_weights=torch.cuda.FloatTensor(compute_class_weight("balanced", np.unique(y_train), y_train))
        # #self.class_weights[0] = self.class_weights[0]/2
        # self.class_weights = torch.cuda.FloatTensor([0.20, 1, 1.2, 1, 1.2])
        # print("weights: {}".format(self.class_weights))
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size*2, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
            nn.Sigmoid(),
        )
        self.size_embeddings = nn.Embedding(self.args.max_span_len, self.config.hidden_size)
        
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
        masked_input_ids = batch.pop("masked_input_ids", None)
        entity_span = batch.pop("entity_span", None)
        entity_mask = batch.pop("entity_mask", None)
        span_len_mask = batch.pop("span_len_mask", None)
        labels = batch.pop("labels", None)

        outputs = self.longformer(
            **batch, 
            global_attention_mask=self._set_global_attention_mask(batch["input_ids"]), output_hidden_states=True
            )

        # cls_output = outputs[0][:, 1]
        sequence_output = outputs[0][:, 1:] 

        ### Span CLF Objective
        ############################################################################################################
        span_index = span_len_mask.view(span_len_mask.size(0), -1).unsqueeze(-1).expand(span_len_mask.size(0), span_len_mask.size(1)*span_len_mask.size(2), sequence_output.size(-1)) # (bs, num_samples*max_span_len, 768)
        span_len_binary_mask = span_len_mask.gt(0).long().unsqueeze(-1)
        span_len_binary_index = span_len_binary_mask.sum(-2).squeeze(-1)
        span_len_embeddings = self.size_embeddings(span_len_binary_index)

        span_len_binary_mask_embeddings = span_len_binary_mask.expand(span_len_binary_mask.size(0), span_len_binary_mask.size(1), span_len_binary_mask.size(2), sequence_output.size(-1))
        span_embeddings = torch.gather(sequence_output, 1, span_index.long()) # (bs, num_samples*max_span_len, 768)
        span_embedding_groups = torch.stack(torch.split(span_embeddings, 15, 1)).transpose(0,1) # (bs, num_samples, max_span_len, 768)
        extracted_spans_after_binary_mask = span_len_binary_mask_embeddings*span_embedding_groups #makes all padding positions into zero vectors (bs, num_samples, max_span_len, 768) 
        maxpooled_embeddings = torch.max(extracted_spans_after_binary_mask, dim=-2).values # (bs, num_samples, 768)

        # if len(cls_output.repeat(sequence_output.size()[0], span_len_mask.size()[1], 1).size())!=len(maxpooled_embeddings.size()):
        #     ipdb.set_trace()

        #combined_embeds = torch.cat([cls_output.unsqueeze(1).repeat(1, 40, 1), maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
        combined_embeds = torch.cat([span_len_embeddings, maxpooled_embeddings], dim=-1) #(bs, num_samples, 2*768)
        logits = self.classifier(combined_embeds).squeeze()        

        ### MLM Objective
        ##############################################################################################################
        prediction_scores = self.lm_head(outputs[0])

        ##############################################################################################################

        total_loss=0
        
        span_clf_loss = None
        if labels!=None:
            loss_fct = nn.BCELoss()
            span_clf_loss = loss_fct(logits.view(-1), labels.view(-1))
            total_loss+=span_clf_loss

        mlm_loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = mlm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), batch["input_ids"].view(-1))
        total_loss+=masked_lm_loss

        return (total_loss, logits, span_clf_loss, masked_lm_loss)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _, span_clf_loss, masked_lm_loss = self(**batch)
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
        loss, logits, span_clf_loss, masked_lm_loss = self(**batch)
        clearlogger.report_text(logits)
        clearlogger.report_scalar(title='mlm_loss', series = 'val', value=masked_lm_loss, iteration=batch_idx) 
        preds = (logits>0.5).long()
        return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_preds = torch.stack([x["preds"] for x in outputs]).view(-1).cpu().detach().tolist()
        val_labels = torch.stack([x["labels"] for x in outputs]).view(-1).cpu().detach().tolist()

        logs = {
            "val_loss": val_loss_mean,
        }
        precision, recall, f1, support = precision_recall_fscore_support(val_labels, val_preds, average='macro')

        self.log("val_loss", logs["val_loss"])
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
            # if idx%2==0:
            #     parameters.requires_grad=False
            # else:
            #     parameters.requires_grad=True
            if idx<6:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
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
