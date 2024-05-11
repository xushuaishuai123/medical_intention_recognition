import json
import random
import torch
from config import Args
from transformers import BertTokenizer
class InputExample:
    def __init__(self,set_type,sentence,seq_label):
        self.set_type=set_type
        self.sentence=sentence
        self.seq_label=seq_label


class InputFeature:
    def __init__(self,input_ids,attention_mask,token_type_ids,seq_label_ids):
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.token_type_ids=token_type_ids
        self.seq_label_ids=seq_label_ids

class Processor:
    @classmethod
    def get_examples(cls,path,set_type):
        raw_examples=[]
        with open(path,'r',encoding='utf-8') as f:
            data=eval(f.read())
        for values in data.values():
            for value in values:
                sentence=value['speaker']+':'+value['sentence']
                dialogue_act=value['dialogue_act']
                raw_examples.append(InputExample(set_type=set_type,
                                                 sentence=sentence,
                                                 seq_label=dialogue_act))
        return raw_examples

def convert_example_to_feature(example,tokenizer,config):
    sentence = example.sentence
    seq_label = example.seq_label
    seq_label_ids=config.seqlabel2id[seq_label]

    inputs = tokenizer.encode_plus(
        text=sentence,
        max_length=config.max_len,
        padding='max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids=torch.tensor(inputs['input_ids'],requires_grad=False)
    attention_mask=torch.tensor(inputs['attention_mask'],requires_grad=False)
    token_type_ids=torch.tensor(inputs['token_type_ids'],requires_grad=False)
    seq_label_ids=torch.tensor(seq_label_ids,requires_grad=False)
    return InputFeature(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        seq_label_ids=seq_label_ids)

def get_features(raw_examples,tokenizer,config):
    features=[]
    for i,example in enumerate(raw_examples):
        feature=convert_example_to_feature(example,tokenizer,config)
        features.append(feature)
    return features

if __name__=='__main__':
    args=Args()
    raw_examples=Processor.get_examples('./data/IMCS-DAC_train.json',set_type='train')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    features = get_features(raw_examples, tokenizer, args)
    i=0
    for feature in features:
        i+=1
        print(feature.input_ids)
        print(feature.attention_mask)
        print(feature.token_type_ids)
        print(feature.seq_label_ids)
        if i>10:
            break









