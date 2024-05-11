import re

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,AutoTokenizer
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import os
from seqeval.metrics.sequence_labeling import get_entities
from tqdm import tqdm

from config import Args
from models import AutoModelForIntentClassification
from dataset import BertDataset
from preprocess import Processor, get_features
import transformers
import json

class Trainer:
    def __init__(self,model,config):
        self.model=model
        self.config=config
        self.criterion=nn.CrossEntropyLoss()
        # self.optimizer=Adam(self.model.parameters(),self.config.lr)
        self.epoch=self.config.epoch
        self.device=self.config.device if self.config.device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == "constantlr":
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == "warmupconstant":
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == "warmuplinear":
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosine":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosinewithhardrestarts":
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def train(self,train_loader):
        global_steps=0
        total_steps=len(train_loader)*self.epoch
        self.model.train()
        self.best_acc=0
        self.config.eval_steps=self.config.eval_steps if self.config.eval_steps>1 else int(self.config.eval_steps*total_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = Adam(optimizer_grouped_parameters, self.config.lr)
        scheduler = self._get_scheduler(
            optimizer, scheduler='warmupcosine', warmup_steps=total_steps * 0.01, t_total=total_steps
        )
        for epoch in tqdm(range(self.epoch),desc='Epoch',disable=not self.config.show_progress_bar):
            for step,train_batch in enumerate(train_loader):
                global_steps += 1
                for key in train_batch.keys():
                    train_batch[key]=train_batch[key].to(self.device)
                input_ids=train_batch['input_ids']
                attention_mask=train_batch['attention_mask']
                token_type_ids=train_batch['token_type_ids']
                seq_label_ids=train_batch['seq_label_ids']
                seq_output=self.model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids
                                      )
                seq_loss=self.criterion(seq_output,seq_label_ids)
                seq_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                print(f'[train] epoch:{epoch+1} {global_steps}/{total_steps} loss:{seq_loss.item()}')


                # if self.config.eval_steps>0 and global_steps%self.config.eval_steps==0 and self.config.do_eval:
                #     print('evaluating with eval_dataset on steps:{}'.format(global_steps))
                #     acc=self.eval(eval_loader)
                #     if acc>self.best_acc:
                #         self.best_acc=acc
                #         self.best_model=self.model
                #         self.save(self.config.save_dir,'model_{0}.pt'.format(self.config.eval_steps))
                #     self.model.zero_grad()
                #     self.model.train()
            self.save(self.config.save_dir, 'model_{0}.pt'.format(global_steps))




        if self.config.do_save:
            self.save(self.config.save_dir,'model_{0}.pt'.format(total_steps))

    def eval(self, eval_loader):
        self.model.eval()
        seq_preds = []
        seq_trues = []
        total_steps = len(eval_loader)
        with torch.no_grad():
            for step, eval_batch in enumerate(eval_loader):
                print(f'[eval] {step}/{total_steps} :')
                for key in eval_batch.keys():
                    eval_batch[key] = eval_batch[key].to(self.device)
                input_ids = eval_batch['input_ids']
                attention_mask = eval_batch['attention_mask']
                token_type_ids = eval_batch['token_type_ids']
                seq_label_ids = eval_batch['seq_label_ids']
                seq_output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids
                                        )
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_label_ids = seq_label_ids.detach().cpu().numpy()
                seq_label_ids = seq_label_ids.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(seq_label_ids)

        acc, precision, recall, f1 = self.get_metrices(seq_trues, seq_preds)
        report = self.get_report(seq_trues, seq_preds)

        print('意图识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            acc, precision, recall, f1
        ))
        print(report)
        return acc
    def get_metrices(self, trues, preds):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average='micro')
        recall = recall_score(trues, preds, average='micro')
        f1 = f1_score(trues, preds, average='micro')
        return acc, precision, recall, f1

    def get_report(self, trues, preds):
        from sklearn.metrics import classification_report
        report = classification_report(trues, preds)

        return report

    def save(self, save_path, save_name):
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))

    
    def predict(self,test_features):
        self.model.eval()
        with torch.no_grad():
            for feature in test_features:
                for key in feature.keys():
                    feature[key] = feature[key].to(self.device)
                input_ids = feature.input_ids
                attention_mask = feature.attention_mask
                token_type_ids = feature.token_type_ids
                seq_output=self.model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_labels=[self.config.id2seqlabel[id] for id in seq_output]
        return seq_labels
def get_last_checkpoint(checkpoint_dir):
    content=os.listdir(checkpoint_dir)
    pattern=re.compile('^model_(\d+).pt$')
    checkpoints = [
        path
        for path in content
        if pattern.search(path) is not None and os.path.isfile(os.path.join(checkpoint_dir, path))
    ]
    if len(checkpoints)==0:
        return
    return os.path.join(checkpoint_dir,max(checkpoints,key=lambda x:int(pattern.search(x).groups()[0])))

def main():
    args = Args()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    if args.do_train:
        raw_examples = Processor.get_examples(args.train_path, 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    if args.do_eval:
        raw_examples = Processor.get_examples(args.eval_path, 'eval')
        eval_features = get_features(raw_examples, tokenizer, args)
        eval_dataset = BertDataset(eval_features)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batchsize, shuffle=True)

    if args.do_test:
        pass

    model = AutoModelForIntentClassification(args)

    if not args.load_dir and args.load_model:
        last_checkpoint = get_last_checkpoint(args.load_dir)
        model.load_state_dict(torch.load(last_checkpoint), strict=False)

    trainer = Trainer(model, args)

    if args.do_train:
        trainer.train(train_loader)

    if args.do_eval:
        trainer.eval(eval_loader)

    if args.do_predict:
        with open('./data/IMCS-DAC_test.json','r',encoding='utf-8-sig') as f:
            test_datas=eval(f.read())
            for key,value in test_datas.items():
                # res_data=[]
                for data in value:
                    sentence=data['speaker']+':'+data['sentence']
                    inputs = tokenizer.encode_plus(
                        text=sentence,
                        max_length=args.max_len,
                        padding='max_length',
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=True,
                    )
                    input_ids = torch.tensor([inputs['input_ids']], requires_grad=False)
                    attention_mask = torch.tensor([inputs['attention_mask']], requires_grad=False)
                    token_type_ids = torch.tensor([inputs['token_type_ids']], requires_grad=False)
                    seq_output=model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
                    seq_output=seq_output.detach().cpu().numpy()
                    seq_label_id=np.argmax(seq_output,axis=-1)
                    seq_label=args.id2seqlabel[seq_label_id[0]]
                    data['dialogue_act']=seq_label
                #     res_data.append(data)
                # test_datas[key]=res_data

        with open('./predict/IMCS-DAC_test.json','w',encoding='utf-8') as f:
            json.dump(test_datas,f,indent=2)


if __name__=='__main__':
    main()
















