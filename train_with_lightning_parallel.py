import json
import os
from typing import Any

import lightning as L
from pytorch_lightning import loggers
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.optim import Adam
import torch.nn as nn
import transformers
from preprocess import Processor,get_features
from dataset import BertDataset
from torch.utils.data import DataLoader
from config import Args
from models import AutoModelForIntentClassification
from transformers import AutoTokenizer
from lightning.pytorch.profilers import SimpleProfiler

class MyLightningModel(L.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.criterion=nn.CrossEntropyLoss()
    def forward(self, input_ids,token_type_ids,attention_mask) -> Any:
        seq_output=self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        return seq_output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer=Adam(self.parameters(),lr=0.01)
        lr_scheduler=transformers.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=1000,num_training_steps=11510)
        return {'optimizer':optimizer,'lr_scheduler':lr_scheduler}
    def training_step(self, batch,batch_idx) -> STEP_OUTPUT:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        seq_label_ids = batch['seq_label_ids']
        seq_output=self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        loss=self.criterion(seq_output,seq_label_ids)
        self.log('train loss',loss)
        return loss

class MyDataModule(L.LightningDataModule):
    def __init__(self,args,tokenozer):
        super().__init__()
        self.args=args
        self.tokenizer=tokenozer
        # self.prepare_data_per_node=False
        # self._log_hyperparams=True
    def prepare_data(self) -> None:
        pass
    def setup(self,stage:str=None) -> None:
        if stage=='train' or stage is None:
            raw_examples = Processor.get_examples(self.args.train_path, 'train')
            train_features = get_features(raw_examples, self.tokenizer, self.args)
            self.train_dataset=BertDataset(train_features)
            raw_examples = Processor.get_examples(self.args.train_path, 'val')
            val_features = get_features(raw_examples, self.tokenizer, self.args)
            self.val_dataset = BertDataset(val_features)
        if stage=='test' or stage is None:
            raw_examples = Processor.get_examples(self.args.train_path, 'test')
            test_features = get_features(raw_examples, self.tokenizer, self.args)
            self.test_dataset = BertDataset(test_features)
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,batch_size=self.args.batchsize)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,batch_size=self.args.batchsize)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,batch_size=self.args.batchsize)


if __name__=='__main__':
    args=Args()
    tokenizer=AutoTokenizer.from_pretrained(args.bert_dir)
    logger=loggers.TensorBoardLogger('./lightning_logs')
    model=MyLightningModel(AutoModelForIntentClassification(args))
    datamodule=MyDataModule(args,tokenizer)
    datamodule.setup()
    trainer=L.Trainer(accelerator='cpu',
                      devices=1,
                      max_epochs=1,
                      enable_checkpointing=True,
                      default_root_dir='./checkpoints',
                      strategy='ddp',
                      logger=logger,
                      profiler='simple',
                      enable_progress_bar=True)
    # trainer.fit(model=model,
    #             train_dataloaders=datamodel.train_dataloader(),
    #             val_dataloaders=datamodel.val_dataloader())
    trainer.fit(model=model,datamodule=datamodule)






