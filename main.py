import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import math
import random
import re
import argparse
import nltk
import time
from tqdm import tqdm
import os
import pickle
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

path_to_images = 'C:/Users/Yash/PycharmProjects/attendanceSystem/images'

path_to_train = 'C:/Users/Yash/PycharmProjects/attendanceSystem/Dataset/train_df.tsv'

path_to_val = 'C:/Users/Yash/PycharmProjects/attendanceSystem/Dataset/val_df.tsv'

path_to_test = 'C:/Users/Yash/PycharmProjects/attendanceSystem/Dataset/test_df.tsv'

path_to_save_model = 'C:/Users/Yash/PycharmProjects/attendanceSystem/Dataset/saved_models/'

class MSEDataset(Dataset):
    def __init__(self, path_to_data_df, path_to_images, tokenizer, image_transform):
        self.data = pd.read_csv(path_to_data_df, sep='\t', names=['pid', 'text', 'explanation'])
        self.path_to_images = path_to_images
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]

        pid_i = row['pid']
        src_text = row['text']
        target_text = row['explanation']

        max_length = 256
        encoded_dict = tokenizer(
            src_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            add_prefix_space = True
        )
        src_ids = encoded_dict['input_ids'][0]
        src_mask = encoded_dict['attention_mask'][0]

        image_path = os.path.join(self.path_to_images, pid_i+'.jpg')
        img = np.array(Image.open(image_path).convert('RGB'))
        img_inp = self.image_transform(img)


        encoded_dict = tokenizer(
          target_text,
          max_length=max_length,
          padding="max_length",
          truncation=True,
          return_tensors='pt',
          add_prefix_space = True
        )

        target_ids = encoded_dict['input_ids'][0]

        sample = {
            "input_ids": src_ids,
            "attention_mask": src_mask,
            "input_image": img_inp,
            "target_ids": target_ids,
        }
        return sample

    def __len__(self):
        return self.data.shape[0]

class MSEDataModule(pl.LightningDataModule):
    def __init__(self, path_to_train_df, path_to_val_df, path_to_test_df, path_to_images, tokenizer, image_transform, batch_size=16):
        super(MSEDataModule, self).__init__()
        self.path_to_train_df = path_to_train_df
        self.path_to_val_df = path_to_val_df
        self.path_to_test_df = path_to_test_df
        self.path_to_images = path_to_images
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def setup(self, stage=None):
        self.train_dataset = MSEDataset(self.path_to_train_df, self.path_to_images, self.tokenizer, self.image_transform)
        self.val_dataset = MSEDataset(self.path_to_val_df, self.path_to_images, self.tokenizer, self.image_transform)
        self.test_dataset = MSEDataset(self.path_to_test_df, self.path_to_images, self.tokenizer, self.image_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler = RandomSampler(self.train_dataset), batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1)

from transformers import BartTokenizer, BartForConditionalGeneration, BartModel, AdamW, BartConfig, BartPretrainedModel, PreTrainedModel
# !pip install pytorch-transformers
from dataclasses import dataclass
from typing import Optional, Tuple, List
from transformers.file_utils import ModelOutput

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.nn import functional as F

from transformers.file_utils import ModelOutput
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.logits_process import (
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def getClones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class CrossmodalMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, img_model=512, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(img_model, d_model)
        self.k_linear = nn.Linear(img_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        #d_ff is set as default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class CrossmodalEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, img_model=512, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = CrossmodalMultiHeadAttention(heads, d_model, img_model=img_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, text_feats, img_feats, mask):
        x = text_feats
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,img_feats,img_feats))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class CrossmodalEncoder(nn.Module):
    def __init__(self, d_model, img_model=512, heads=4, N=1, dropout=0.1):
        super(CrossmodalEncoder, self).__init__()
        self.N = N
        self.cme_layers = getClones(CrossmodalEncoderLayer(d_model, heads, img_model=img_model, dropout=dropout), N)
        self.norm = Norm(d_model)

    def forward(self, text_feats, img_feats, mask):
        x = text_feats
        for i in range(self.N):
            x = self.cme_layers[i](x, img_feats, mask)
        return self.norm(x)

class MultimodalBartEncoder(PreTrainedModel):
    def __init__(self, bart_encoder, bart_config, image_encoder, img_model=512, N=1, heads=4, dropout=0.1):
        super(MultimodalBartEncoder, self).__init__(bart_config)
        self.config = bart_config
        self.bart_encoder = bart_encoder
        self.image_encoder = image_encoder
        self.N=N
        self.img_model = img_model
        self.cross_modal_encoder = CrossmodalEncoder(self.config.d_model, img_model=img_model, heads=heads, N=N, dropout=dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        image_features=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            vgg_image_features = self.image_encoder(image_features)

            vgg_image_features = vgg_image_features.permute(0, 2, 3, 1)
            vgg_image_features = vgg_image_features.view(
                -1,
                vgg_image_features.size()[1]*vgg_image_features.size()[2],
                self.img_model
                )

            encoder_outputs = self.bart_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            cross_modal_encoder_outputs = self.cross_modal_encoder(
                encoder_outputs.last_hidden_state,
                vgg_image_features,
                attention_mask
            )

            encoder_outputs.last_hidden_state = torch.cat((encoder_outputs.last_hidden_state, cross_modal_encoder_outputs), dim=-2)
            return encoder_outputs

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class BartForMultimodalSarcasmDetection(BartPretrainedModel):
    def __init__(self, bart_model_encoder, bart_config, image_encoder, num_labels=2, dropout_rate=0.1, img_model=512, N=1, heads=4):
        super(BartForMultimodalSarcasmDetection, self).__init__(bart_config)
        self.config = bart_config
        self.encoder = MultimodalBartEncoder(bart_model_encoder, bart_config, image_encoder, img_model=img_model, N=N, heads=heads, dropout=dropout_rate)
        self.classification_head = BartClassificationHead(
            self.config.d_model,
            self.config.d_model,
            num_labels,
            dropout_rate,
        )
        self._init_weights(self.classification_head.dense)
        self._init_weights(self.classification_head.out_proj)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        image_features = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        classification_feature_vector = encoder_outputs.last_hidden_state.mean(dim=-2)
        logits = self.classification_head(classification_feature_vector)
        loss = None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.last_hidden_state,
            attentions=encoder_outputs.attentions,
        )

class PyLitModel(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.hparams.update(hparams)

        if self.hparams['freeze_encoder']:
            freeze_params(self.model.encoder.bart_encoder)

        if self.hparams['freeze_embeds']:
            self.freeze_embeds()

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.bart_model_shared)
        for d in [self.model.encoder.bart_encoder, self.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.cross_modal_encoder.parameters(), "lr": self.hparams['lr']},
                {"params": self.model.classification_head.parameters(), "lr": self.hparams['lr']},
            ],
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        src_ids, src_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        image_features = batch['input_image'].to(device)
        labels = batch['target_ids'].to(device)

        outputs = self(src_ids, attention_mask=src_mask, image_features=input_images, use_cache=False)
        classification_logits = outputs.logits

        # The loss function
        ce_loss = torch.nn.CrossEntropyLoss() #ignore_index=self.tokenizer.pad_token_id)

        # Calculate the loss on the un-shifted tokens
        loss = ce_loss(classification_logits.view(-1, classification_logits.shape[-1]), labels.view(-1))

        self.log('train_cross_entropy_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):

        src_ids = batch['input_ids'].to(device)
        src_mask = batch['attention_mask'].to(device)
        image_features = batch['input_image'].to(device)
        labels = batch['target_ids'].to(device)

        outputs = self(src_ids, attention_mask=src_mask, image_features=input_images, use_cache=False)
        classification_logits = outputs.logits

        ce_loss = torch.nn.CrossEntropyLoss() #ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss(classification_logits.view(-1, classification_logits.shape[-1]), labels.view(-1))

        self.log('val_cross_entropy_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_score', f1(F.softmax(classification_logits, dim=1), labels, num_classes=2), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': val_loss}

    def predict(self, src_ids, src_mask, input_images):
        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)
        input_images = input_images.to(device)

        outputs = self(src_ids, attention_mask=src_mask, input_images=input_images, use_cache=False)
        classification_logits = outputs.logits
        class_probs = F.softmax(classification_logits, dim=1)
        return torch.argmax(class_probs, dim=1)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartForMultimodalSarcasmExplanation(BartPretrainedModel):
    def __init__(self, multimodal_bart_encoder_TL, bart_decoder, bart_config, bart_model_num_embs, img_model=512, N=1, heads=4):
        super(BartForMultimodalSarcasmExplanation, self).__init__(bart_config)
        self.config = bart_config
        self.encoder = multimodal_bart_encoder_TL
        self.decoder = bart_decoder
        self.lm_head = nn.Linear(self.config.d_model, bart_model_num_embs) #, bias=False)

        self._init_weights(self.lm_head)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids, past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        image_features=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "image_features": image_features,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    #def adjust_logits_during_generation(self, logits, cur_len, max_length):
    #    if cur_len == 1 and self.config.force_bos_token_to_be_generated:
    #        self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
    #    elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
    #        self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
    #    return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,

        image_features = None,

        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        enc_attn_mask = torch.cat((attention_mask, attention_mask), dim=-1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=enc_attn_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)

        masked_lm_loss = None
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state, # also carries crossmodal_encoder_last_hidden_state concatenated.
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class PyLitBartForMultimodalSarcasmExplanation(pl.LightningModule):
    def __init__(self, model, tokenizer, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.hparams.update(hparams)

        if self.hparams['freeze_image_encoder']:
            freeze_params(self.model.encoder.image_encoder)

        if self.hparams['freeze_encoder']:
            freeze_params(self.model.encoder.bart_encoder)

        if self.hparams['freeze_embeds']:
            self.freeze_embeds()

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.bart_model_shared)
        for d in [self.model.encoder.bart_encoder, self.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
          [
              {"params": self.model.encoder.cross_modal_encoder.parameters(), "lr": self.hparams['lr_finetune_cm']},
              {"params": self.model.lm_head.parameters(), "lr": self.hparams['lr']},
          ],
        )
        return optimizer

    def training_step(self, batch, batch_idx):

        src_ids, src_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        image_features = batch['input_image'].to(device)
        tgt_ids = batch['target_ids'].to(device)

        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, image_features=image_features, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs.logits

        # the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        src_ids = batch['input_ids'].to(device)
        src_mask = batch['attention_mask'].to(device)
        image_features = batch['input_image'].to(device)
        tgt_ids = batch['target_ids'].to(device)

        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, image_features=image_features, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs.logits

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': val_loss}

    # This method generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, text, eval_beams, image_features=None, early_stopping = True, max_len = 40):
        ''' Function to generate text '''

        model_kwargs = {
            "image_features": image_features
        }
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            early_stopping = early_stopping,
            **model_kwargs,
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

def freeze_params(model):
    ''' This function takes a model or its subset as input and freezes the layers for faster training
      adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False

def load_image_encoder():
    vgg19model = models.vgg19(pretrained=True)
    image_encoder = list(vgg19model.children())[0]
    return image_encoder

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)

bart_model = BartModel.from_pretrained('facebook/bart-base')

bart_config = BartConfig.from_pretrained("facebook/bart-base", return_dict=True)

image_encoder = load_image_encoder()

hparams = {
    'freeze_encoder': False,
    'freeze_embeds': False,
    'freeze_image_encoder': True,
    'eval_beams': 4,
    'lr_finetune_cm':1e-5, #for crossmodal encoder
    'lr': 3e-4, #for lm_head
}

bart_model_for_msd = BartForMultimodalSarcasmDetection(
    bart_model.get_encoder(),
    bart_config,
    image_encoder,
    num_labels=2,
    dropout_rate=0.1,
    img_model=512,
    N=1,
    heads=4,
)

msd_checkpoint_path = 'C:/Users/Yash/PycharmProjects/attendanceSystem/MSD_pretrained_model.ckpt'
pylit_bart_model_for_msd = PyLitModel.load_from_checkpoint(checkpoint_path=msd_checkpoint_path,
                                      model = bart_model_for_msd,
                                      hparams = hparams)

multimodal_bart_encoder_TL = pylit_bart_model_for_msd.model.get_encoder()
bart_decoder = bart_model.get_decoder()
bart_model_num_embs = bart_model.shared.num_embeddings

bart_for_mse = BartForMultimodalSarcasmExplanation(multimodal_bart_encoder_TL,
                                            bart_decoder, bart_config,
                                            bart_model_num_embs, img_model=512, N=1, heads=4)

ckpt_path = 'C:/Users/Yash/PycharmProjects/attendanceSystem/ExMore_model.ckpt'
main_model = PyLitBartForMultimodalSarcasmExplanation.load_from_checkpoint(checkpoint_path=ckpt_path,
                                      tokenizer = tokenizer, model = bart_for_mse, hparams = hparams)
test = pd.read_csv(path_to_test, sep='\t', header=None)
# test = pd.read_csv(path_to_train, sep='\t', header=None)
test.columns = ['pid', 'source', 'target']
pids = test.pid.tolist()
source = test.source.tolist()
target = test.target.tolist()

main_model.to(device)
main_model.eval()

eval_beams=4
pred = []

for pid_i, src, tgt in tqdm(zip(pids, source, target)):
    encoded_dict = tokenizer(
      src,
      max_length=256,
      padding="max_length",
      truncation=True,
      return_tensors='pt',
      add_prefix_space = True
    )
    encoded_dict['input_ids'] = encoded_dict['input_ids'].to(device)
    encoded_dict['attention_mask'] = encoded_dict['attention_mask'].to(device)

    if type(pid_i) is not str:
        pid_i = str(pid_i)

    image_path = os.path.join(path_to_images, pid_i+'.jpg')
    img = np.array(Image.open(image_path).convert('RGB'))
    img_feats = image_transform(img)
    img_feats = img_feats.unsqueeze(0)

    gen = main_model.generate_text(
      encoded_dict,
      eval_beams,
      image_features=img_feats.to(device),
      early_stopping = True,
      max_len = 256
    )

    pred.append(gen[0])
    hypothesis = gen[0].split()
    reference = tgt.split()

predictions = pd.DataFrame({0:pred})

path_to_predictions = 'C:/Users/Yash/PycharmProjects/attendanceSystem/Outputs/prediction.csv'
predictions.to_csv(path_to_predictions, sep='\t', index=False, header=False)