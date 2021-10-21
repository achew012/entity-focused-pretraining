from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math, ipdb

# Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='ner-pretraining', task_name='MLM+BIO-Loss', output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

# config = json.load(open('config.json'))

config={
    "lr": 3e-4,
    "num_epochs":50,
    "train_batch_size":1,
    "eval_batch_size":1,
    "max_length": 2048, # be mindful underlength will cause device cuda side error
    "use_entities_as_spans": True, # Toggle between entity pretraining or span pretraining
    "mlm_task": False,
    "sbo_task": False,
    "bio_task": True,
}
args = argparse.Namespace(**config)
# task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
# task.execute_remotely(queue_name="128RAMv100", exit_process=True)
# task.connect(args)

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

class DWIE_Data(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.consolidated_dataset = []
        self.max_spans = 30 # 15 -> 95 , 25 -> 88, 20 -> 92 docs
        self.class2id = {"B":0, "I":1, "O":2}
        
        if self.args.use_entities_as_spans:
            print("Using Entities as Spans")
        else:
            print("Using Random Continuous Tokens as Spans")

        for doc in self.dataset:
            if len(doc["entities"])>0:
                context = ' '.join(doc["sentences"])
                self.encodings = self.tokenizer(context, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
                masked_input_ids = self.encodings["input_ids"].clone()            
                doc_entity_mask = torch.zeros(self.tokenizer.model_max_length)
                sbo_labels = torch.zeros(self.tokenizer.model_max_length)
                bio_labels = torch.full_like(self.encodings["input_ids"], self.class2id["O"])
                entity_spans = []

                # Use entities as spans
                if self.args.use_entities_as_spans:
                    entities_samples = random.choices(doc["entities"], k=self.max_spans) # with replacement
                    #entities_samples = random.sample(doc["entities"], self.max_spans) # without replacement

                    for idx, entity in enumerate(entities_samples):
                        entity_start = entity["start"] if entity["start"] < self.tokenizer.model_max_length else 0
                        entity_end = entity["end"] if entity["end"]<self.tokenizer.model_max_length else self.tokenizer.model_max_length-1

                        # Mask entities in the document
                        # masked_input_ids[:, entity_start:entity_end] = torch.tensor([random.randint(10, self.tokenizer.vocab_size-10) for sample in masked_input_ids for entity in sample[entity_start:entity_end]]) if random.random()>0.85 else self.tokenizer.mask_token_id
                        masked_input_ids[:, entity_start:entity_end] = self.tokenizer.mask_token_id                    

                        # Get BIO Labels
                        bio_labels[:, :entity_start+1] = self.class2id["B"]
                        if entity_end!=entity_start+1:
                            bio_labels[:, entity_start+1:entity_end] = self.class2id["I"]

                        # get mask of entities
                        doc_entity_mask[entity_start:entity_end] = idx 

                        # Specify entity boundary positions
                        entity_spans.append(torch.tensor((entity_start-1 if entity_start>0 else 0, entity_end)))


                # Use random contiguous spans
                else:
                    for idx in range(self.max_spans):
                        entity_start = random.randint(5, len(self.tokenizer.tokenize(context))-15)
                        entity_end = random.randint(entity_start, entity_start+random.randint(1, 10))

                        # mask entities in the document
                        masked_input_ids[:, entity_start:entity_end] = torch.tensor([random.randint(10, self.tokenizer.vocab_size-10) if random.random()>0.85 else self.tokenizer.mask_token_id for sample in masked_input_ids for entity in sample[entity_start:entity_end]])
                        #masked_input_ids[:, entity_start:entity_end] = self.tokenizer.mask_token_id                    
                        bio_labels[:, :entity_start+1] = self.class2id["B"]
                        bio_labels[:, entity_start+1:entity_end] = self.class2id["I"]

                        # get mask of entities
                        doc_entity_mask[entity_start:entity_end] = idx 

                        # Specify entity boundary positions
                        entity_spans.append(torch.tensor((entity_start-1 if entity_start>0 else 0, entity_end)))

                self.consolidated_dataset.append({
                    "masked_input_ids": masked_input_ids,
                    "input_ids": self.encodings["input_ids"],
                    "attention_mask": self.encodings["attention_mask"],
                    "entity_span": torch.stack(entity_spans),
                    "entity_mask": doc_entity_mask,
                    "bio_labels": bio_labels
                    # "entity_start": entity_start,
                    # "entity_end": entity_end,
                })           

        # ipdb.set_trace()

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
        bio_labels = torch.stack([ex['bio_labels'] for ex in batch])
        # entity_start = torch.stack([torch.tensor(ex['entity_start']) for ex in batch]).unsqueeze(1) 
        # entity_end = torch.stack([torch.tensor(ex['entity_end']) for ex in batch]).unsqueeze(1) 
        
        return {
                "masked_input_ids": masked_input_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "entity_span": entity_span,
                "entity_mask": entity_mask,
                "bio_labels": bio_labels,
                # "entity_start": entity_start,
                # "entity_end": entity_end,
        }

class NERLongformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        self.config.gradient_checkpointing = True
        self.config.num_labels = 3

        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer.model_max_length = self.args.max_length
        self.longformer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True)

        train_data = DWIE_Data(dwie["train"], self.tokenizer, self.args)
        y_train = torch.stack([doc["bio_labels"] for doc in train_data.consolidated_dataset]).view(-1).cpu().numpy()
        self.class_weights=torch.cuda.FloatTensor(compute_class_weight("balanced", np.unique(y_train), y_train))
        print("weights: {}".format(self.class_weights))

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)       
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.softmax = nn.Softmax(dim=-1)

        #self.longformer = LongformerModel(self.config)
        #self.lm_head = LongformerLMHead(self.config) 

        # SBO representation
        self.sbo = nn.Sequential(
          nn.Linear(3*self.config.hidden_size, self.config.hidden_size), # 3 =  start + end + position embeddings,  output is arbitrary
          nn.GELU(),
          nn.LayerNorm(self.config.hidden_size),
          nn.Linear(self.config.hidden_size, self.config.vocab_size),
          nn.GELU(),
          nn.LayerNorm(self.config.vocab_size),
        )
        
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

        masked_input_ids, input_ids, attention_mask, entity_span, entity_mask = batch["masked_input_ids"], batch["input_ids"], batch["attention_mask"], batch["entity_span"], batch["entity_mask"]
        
        mlm_labels = input_ids
        sbo_labels = torch.masked_select(input_ids, entity_mask.gt(0))

        if "bio_labels" in batch.keys():
            bio_labels = batch["bio_labels"]
        else:
            bio_labels = None

        ## loss, logits for MLM
        masked_lm_loss = None        
        if self.args.mlm_task:
            outputs = self.longformer(
                input_ids = masked_input_ids,
                attention_mask = attention_mask,
                global_attention_mask=self._set_global_attention_mask(masked_input_ids),
                labels = mlm_labels,
                output_hidden_states=True
            )
            
            sequence_output = outputs.hidden_states[-1]
            prediction_scores = outputs.logits
            masked_lm_loss = outputs.loss  

        ## loss, logits for Span Boundary Objective
        sbo_scores=None
        if self.args.sbo_task:
            outputs = self.longformer(
                input_ids = masked_input_ids,
                attention_mask = attention_mask,
                global_attention_mask=self._set_global_attention_mask(masked_input_ids),
                labels = mlm_labels,
                output_hidden_states=True
            )
            
            #sequence_output = outputs[0]
            sequence_output = outputs.hidden_states[-1]

            # get boundary pair indices to look up embedding
            span_index = entity_span.view(entity_span.size(0), -1).unsqueeze(-1).expand(entity_span.size(0), entity_span.size(1)*entity_span.size(2), sequence_output.size(-1))
            # Lookup from output of longformer
            span_embeddings = torch.gather(sequence_output, 1, span_index) #
            # group into entity boundary pairs
            span_embedding_pairs = torch.stack(torch.split(span_embeddings, 2, 1)).squeeze() # Number of Entity Boundary Pair Embeddings - (bs * N * 2 * 768)
            # get entity positional indices from mask
            entity_boundaries = torch.masked_select(entity_mask, entity_mask.gt(0)).long()
            # replace positional indices with lookup from span boundary embedding pairs
            span_embedding_sequence = torch.index_select(span_embedding_pairs, 0, entity_boundaries).squeeze() # Seq
            # Get positional indices of masked entity
            pos_idx = torch.stack([((sample > 0).nonzero(as_tuple=False)).squeeze(-1) for sample in entity_mask])
            # Look up positional embeddings
            #masked_token_position_embeddings = self.longformer.embeddings.position_embeddings(pos_idx).view(-1, 1, sequence_output.size(-1))        
            masked_token_position_embeddings = self.longformer.longformer.embeddings.position_embeddings(pos_idx).view(-1, 1, sequence_output.size(-1))        

            if len(span_embedding_sequence.size())<3:
                span_embedding_sequence = span_embedding_sequence.unsqueeze(0)

            if len(masked_token_position_embeddings.size())<3:
                masked_token_position_embeddings = masked_token_position_embeddings.unsqueeze(0)

            # Combine x_start, x_end and token_embedding_i to single dim
            combined_representation = torch.cat((span_embedding_sequence, masked_token_position_embeddings), 1).view(-1, 3*sequence_output.size(-1))
            # pass through spanbert sbo representation
            sbo_scores = self.sbo(combined_representation)

        # BIO Classification Objective
        bio_scores=None
        if self.args.bio_task:
            outputs = self.longformer(
                input_ids = input_ids,
                attention_mask = attention_mask,
                global_attention_mask=self._set_global_attention_mask(input_ids),
                output_hidden_states=True
            )
            sequence_output = outputs.hidden_states[-1]
            bio_output = self.dropout(sequence_output)
            bio_scores = self.classifier(bio_output)

        # span_loss = None
        # bio_loss = None
        total_loss = 0
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        if mlm_labels is not None and masked_lm_loss!=None:
            #masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            total_loss+=masked_lm_loss
        if sbo_labels is not None and sbo_scores!=None:
            span_loss = loss_fct(sbo_scores.view(-1, self.config.vocab_size), sbo_labels.view(-1))
            total_loss+=span_loss
        if bio_labels is not None and bio_scores!=None:
            loss_fct.weight=self.class_weights
            bio_loss = loss_fct(bio_scores.view(-1, self.config.num_labels), bio_labels.view(-1))
            total_loss+=bio_loss

        return (total_loss, bio_scores)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _ = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
            "train_perplexity": torch.exp(train_loss_mean)
        }

        self.log("train_loss", logs["train_loss"])
        self.log("train_perplexity", logs["train_perplexity"])

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        preds = torch.argmax(self.softmax(logits), dim=-1)
        # print(hidden_states)
        return {"val_loss": loss, "preds": preds, "labels": batch["bio_labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_preds = torch.stack([x["preds"] for x in outputs]).view(-1).cpu().detach().tolist()
        val_labels = torch.stack([x["labels"] for x in outputs]).view(-1).cpu().detach().tolist()

        print(classification_report(val_preds, val_labels))

        logs = {
            "val_loss": val_loss_mean,
            "val_perplexity": torch.exp(val_loss_mean)
        }

        self.log("val_loss", logs["val_loss"])
        self.log("val_perplexity", logs["val_perplexity"])

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
            if idx%2==0:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.longformer.parameters(), lr=self.args.lr)
        return optimizer

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_entity_lm", 
    monitor="val_perplexity", 
    mode="min", 
    save_top_k=1, 
    save_weights_only=True
)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=8, verbose=False, mode="min")
NERLongformer = NERLongformer(args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(NERLongformer)
