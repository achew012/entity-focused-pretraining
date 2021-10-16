from clearml import Task, StorageManager, Dataset as ds
import argparse, json, os, random, math

# Task.add_requirements('transformers', package_version='4.2.0')
task = Task.init(project_name='ner-pretraining', task_name='NER-LM', output_uri="s3://experiment-logging/storage/")

# config = json.load(open('config.json'))
# args = argparse.Namespace(**config)
# task.connect(args)

task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.execute_remotely(queue_name="128RAMv100", exit_process=True)
clearlogger = task.get_logger()

dataset = ds.get(dataset_name="processed-DWIE", dataset_project="datasets/DWIE", dataset_tags=["1st-mention"], only_published=True)
dataset_folder = dataset.get_local_copy()
dwie = json.load(open(os.path.join(dataset_folder, "data", "new_dwie.json")))["dataset"]

import torch
from transformers import LongformerForMaskedLM, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader

def get_entities_mask(entities_list, max_seq_length=1024, mask_size = 0.5):
    entities_masks = []
    for entities in entities_list:
        #randomly select subset of entities
        entities_subset = random.sample(entities, math.ceil(mask_size*len(entities)))
        #create a mask of the positions of these 0.1 entities relative to max length
        doc_entity_mask = torch.zeros(max_seq_length)
        for start, end in entities_subset:
            # +1 to cater for <CLS>
            start, end = start+1, end+1
            if end<max_seq_length:
                doc_entity_mask[start:end]=1
        entities_masks.append(doc_entity_mask)
    entities_masks = torch.stack(entities_masks)
    return entities_masks.gt(0)

class DWIE_Data(Dataset):
    def __init__(self, documents, entities, tokenizer):
        self.documents = documents
        self.entities = entities
        self.tokenizer = tokenizer
        
        max_length=1024

        #self.encodings = self.tokenizer(self.documents, padding=True, truncation=True, return_tensors="pt")
        #self.encodings["input_ids"]=self.mask_entities(entities, self.encodings["input_ids"])

        input_ids = [self.tokenizer.encode(doc) for doc in self.documents]
        input_ids = torch.tensor([tokens+(max_length-len(tokens))*[self.tokenizer.pad_token_id] if len(tokens)<max_length else tokens[:max_length] for tokens in input_ids])
        attention_mask = ~(input_ids == self.tokenizer.pad_token_id)
        input_ids=self.mask_entities(entities, input_ids)
        decoder_input_ids = input_ids
        decoder_attention_mask = attention_mask
        self.encodings ={
            "input_ids":input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        self.labels = self.encodings["input_ids"]

    def mask_entities(self, entities, input_ids):
        entities_mask  = get_entities_mask(entities, max_seq_length=input_ids.size()[-1])
        # Get input ids and replace with mask token
        #input_ids[entities_mask] = self.tokenizer.mask_token_id
        input_ids[entities_mask] = torch.tensor([random.randint(40, self.tokenizer.vocab_size-10) if random.random()>0.9 else self.tokenizer.mask_token_id for entity in input_ids[entities_mask]])
        return input_ids

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
        decoder_input_ids = torch.stack([ex['decoder_input_ids'] for ex in batch]) 
        decoder_attention_mask = torch.stack([ex['decoder_attention_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch]) 

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


# tokenizer = LongformerTokenizer.from_pretrained("allenai/led-base-16384")
# train = dwie["train"]
# train_sentences = [' '.join(doc["sentences"]) for doc in train]
# train_entities = [[(entity["start"], entity["end"]) for entity in doc["entities"]] for doc in train]

# max_length = 4096
# input_ids = [tokenizer.encode(doc) for doc in train_sentences]
# input_ids = torch.tensor([tokens+(max_length-len(tokens))*[tokenizer.pad_token_id] for tokens in input_ids])

# input_ids = tokenizer(train_sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
# entities_mask  = get_entities_mask(train_entities, input_ids.size()[-1], mask_size = 0.3)
# original_input_ids = input_ids.clone()
# input_ids[entities_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
# sample1 = tokenizer.decode(original_input_ids[0])
# sample2 = tokenizer.decode(input_ids[0])

# data=DWIE_Data(train_sentences, train_entities, tokenizer)

# import ipdb; ipdb.set_trace()
# import sys; sys.exit()


import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import LongformerConfig, LEDConfig, LEDTokenizer, LEDForConditionalGeneration, LEDModel
from transformers.models.longformer.modeling_longformer import LongformerLMHead


class NERLongformer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.args = params
        #self.softmax = nn.Softmax(dim=-1)

        model_name = "allenai/led-base-16384"         
        self.config = LEDConfig.from_pretrained(model_name)
        # self.config.max_encoder_position_embeddings = 4096
        # self.config.max_decoder_position_embeddings = 4096
        self.config.gradient_checkpointing=True

        self.tokenizer = LEDTokenizer.from_pretrained(model_name, config=self.config)
        self.model = LEDModel.from_pretrained(model_name, config=self.config)
        lm_config= LongformerConfig.from_pretrained("allenai/longformer-base-4096")
        self.lm_head = LongformerLMHead(lm_config)

    def val_dataloader(self):
        val = dwie["test"]
        val_sentences = [' '.join(doc["sentences"]) for doc in val]
        val_entities = [[(entity["start"], entity["end"]) for entity in doc["entities"]] for doc in val]
        val_data = DWIE_Data(val_sentences, val_entities, self.tokenizer)
        val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
        return val_dataloader

    def train_dataloader(self):
        train = dwie["train"]
        train_sentences = [' '.join(doc["sentences"]) for doc in train]
        train_entities = [[(entity["start"], entity["end"]) for entity in doc["entities"]] for doc in train]
        train_data = DWIE_Data(train_sentences, train_entities, self.tokenizer)
        train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, collate_fn = train_data.collate_fn)
        return train_dataloader

    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions
        labels = batch.pop("labels", None)

        # import ipdb; ipdb.set_trace()
        outputs = self.model(**batch)
        sequence_output = outputs[0] 
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return masked_lm_loss, prediction_scores

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # print(hidden_states)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        logs = {
            "val_loss": val_loss_mean,
            "val_perplexity": torch.exp(val_loss_mean)
        }

        # clearlogger.report_scalar(title='perplexity', series = 'val_perplexity', value=logs["val_perplexity"], iteration=self.trainer.current_epoch) 

        self.log("val_loss", logs["val_loss"])
        self.log("val_perplexity", logs["val_perplexity"])

    # Freeze weights?
    def configure_optimizers(self):

        # Freeze 1st 6 layers of longformer
        for idx, (name, parameters) in enumerate(self.model.named_parameters()):
            if idx<6:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./",
    filename="best_entity_lm", 
    monitor="val_perplexity", 
    mode="min", 
    save_top_k=1, 
    # save_weights_only=True
)

args={
    "num_epochs":100,
    "train_batch_size":6,
    "eval_batch_size":4
}

class bucket_ops:
    StorageManager.set_cache_file_limit(5, cache_context=None)

    def list(remote_path:str):
        return StorageManager.list(remote_path, return_full_path=False)

    def upload_folder(local_path:str, remote_path:str):
        StorageManager.upload_folder(local_path, remote_path, match_wildcard=None)
        print("Uploaded {}".format(local_path))

    def download_folder(local_path:str, remote_path:str):
        StorageManager.download_folder(remote_path, local_path, match_wildcard=None, overwrite=True)
        print("Downloaded {}".format(remote_path))
    
    def get_file(remote_path:str):        
        object = StorageManager.get_local_copy(remote_path)
        return object

    def upload_file(local_path:str, remote_path:str):
        StorageManager.upload_file(local_path, remote_path, wait_for_upload=True, retries=3)

# trained_model_path = bucket_ops.get_file(
#     remote_path="s3://experiment-logging/storage/ner-pretraining/NER-LM.7aa2cc034cdf4b619b1dbf9ffffac9b0/models/best_entity_lm.ckpt"
#     )

args = argparse.Namespace(**args)
model = NERLongformer(args)
#model = NERLongformer.load_from_checkpoint(trained_model_path, params = args)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, callbacks=[checkpoint_callback])
trainer.fit(model)
