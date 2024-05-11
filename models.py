from typing import Callable

from torch.utils.hooks import RemovableHandle
from transformers import BertTokenizer, BertModel,AutoModel
import torch.nn as nn
import torch
class AutoModelForIntentClassification(nn.Module):
    def __init__(self,config):
        super(AutoModelForIntentClassification,self).__init__()
        self.config=config
        self.automodel=AutoModel.from_pretrained(self.config.bert_dir)

        self.sequence_classification=nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size,self.config.seq_num_labels)
        )

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids):


        automodel_output=self.automodel(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         )
        # token_output=automodel_output[1]
        # token_output=torch.sum(token_output,dim=1)
        # attention_mask_expand=attention_mask.unsqueeze(-1).expand(automodel_output[0].size()).float()
        # sum_mask_expand=torch.sum(attention_mask_expand,dim=1)
        # sum_mask_expand=torch.clamp(sum_mask_expand,min=1e-9)
        output=automodel_output[1]


        seq_output=self.sequence_classification(output)
        return seq_output

