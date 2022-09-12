
# coding: utf-8

# In[1]:


import torch
from transformers import *


# In[2]:


import json
import sys
sys.path.insert(0,'../')

from konlpy.tag import Kkma

import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# In[3]:


from teddy_srl import srl_model
from teddy_srl import preprocessor
from teddy_srl import conll2textae


# In[4]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[5]:


def remove_josa(phrase):
    from konlpy.tag import Kkma
    kkma = Kkma()
    import jpype
    jpype.attachThreadToJVM()
    
    tokens = phrase.split(' ')

    result = []
    for i in range(len(tokens)):
        token = tokens[i]
        if i < len(tokens)-1:
            result.append(token)
        else:
            m = kkma.pos(tokens[i])
            if m[-1][-1].startswith('J'):
                josa = m[-1][0]                
                josa_index = token.rfind(josa)
                token = token[:josa_index]
            result.append(token)
    result = ' '.join(result)
    return result


# In[6]:


def conll2graph(conll, sent_id=False, language='ko'):
    triples = []
    n = 0
    for anno in conll:
        tokens, targets, args = anno[0],anno[1],anno[2]
        
        target_id = -1
        for i in range(len(targets)):
            t = targets[i]
            if t != '_':
                target_id = i
                target = t
                
#         if target:
#             if type(sent_id) != bool:
#                 triple = ('predicate'+'#'+str(sent_id)+'-'+str(n)+':'+frame, 'lu', lu_token)
#                 triples.append(triple)
#             else:
#                 triple = ('predicate'+'#'+str(n)+':'+frame, 'lu', lu_token)
#                 triples.append(triple)

        if target:
            sbj = False
            pred_obj_tuples = []
            
            for idx in range(len(args)):
                arg_tag = args[idx]
                arg_tokens = []
                if arg_tag.startswith('B'):
                    fe_tag = arg_tag.split('-')[1]
                    arg_tokens.append(tokens[idx])
                    next_idx = idx + 1
                    while next_idx < len(args) and args[next_idx] == 'I-'+fe_tag:
                        arg_tokens.append(tokens[next_idx])
                        next_idx +=1
                    arg_text = ' '.join(arg_tokens)
                    
                    if language =='ko':
                        arg_text = remove_josa(arg_text)
                    else:
                        pass
                    fe = fe_tag
                    
                    rel = 'arg:'+fe

                    if rel == 'S':
                        pass
                    else:
                        p = rel
                        o = arg_text
                        pred_obj_tuples.append( (p,o) )

            for p, o in pred_obj_tuples:
                if sbj:
                    s = sbj
                else:
                    if type(sent_id) != bool:
                        s = 'predicate'+'#'+str(sent_id)+'-'+str(n)+':'+target
                    else:
                        s = 'predicate'+'#'+str(n)+':'+target
                triple = (s, p, o)
                triples.append(triple)
        n +=1
    return triples


# In[7]:


class srl_parser():
    
    def __init__(self, model_dir=False, batch_size=1):
#         try:
        self.model = BertForTokenClassification.from_pretrained(model_dir)
        self.model.to(device);
        self.model.eval()
#         except KeyboardInterrupt:
#             raise
#         except:
#             print('model dir', model_dir, 'is not valid ')
            
        self.bert_io = srl_model.for_BERT(mode='test')
        self.batch_size = batch_size
        
    def ko_srl_parser(self, text):
        
        input_data = preprocessor.preprocessing(text)        
        input_tgt_data = preprocessor.data2tgt_data(input_data)        
        input_data_bert = self.bert_io.convert_to_bert_input(input_tgt_data)        
        input_dataloader = DataLoader(input_data_bert, batch_size=self.batch_size)
        
        pred_args = []
        for batch in input_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_masks = batch
            
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_masks)
                logits = outputs.logits
                
            logits = logits.detach().cpu().numpy()
            b_pred_args = [list(p) for p in np.argmax(logits, axis=2)]
            
            for b_idx in range(len(b_pred_args)):
                
                input_id = b_input_ids[b_idx]
                orig_tok_to_map = b_input_orig_tok_to_maps[b_idx]                
                pred_arg_bert = b_pred_args[b_idx]

                pred_arg = []
                for tok_idx in orig_tok_to_map:
                    if tok_idx != -1:
                        tok_id = int(input_id[tok_idx])
                        if tok_id == 1:
                            pass
                        elif tok_id == 2:
                            pass
                        else:
                            pred_arg.append(pred_arg_bert[tok_idx])                            
                pred_args.append(pred_arg)
                
        pred_arg_tags_old = [[self.bert_io.idx2tag[p_i] for p_i in p] for p in pred_args]
        
        conll = []
        for b_idx in range(len(pred_arg_tags_old)):
            pred_arg_tag_old = pred_arg_tags_old[b_idx]
            pred_arg_tag = []
            for t in pred_arg_tag_old:
                if t == 'X':
                    new_t = 'O'
                else:
                    new_t = t
                pred_arg_tag.append(new_t)
                
            instance = []
            instance.append(input_data[b_idx][0])
            instance.append(input_data[b_idx][1])
            instance.append(pred_arg_tag)
            
            conll.append(instance)
        
        graph = conll2graph(conll)
        textae = conll2textae.get_textae(conll)
        
        result = {}
        result['conll'] = conll
        result['graph'] = graph
        result['textae'] = textae
        
        
        
        return result

