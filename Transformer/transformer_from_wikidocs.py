import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import pandas as pd
import random
import re

from torch.utils.data import dataloader, Dataset

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_rate=0.1, ):
        """
        Transformer model constructor.

        Args:
        - num_tokens (int): Number of tokens in the vocabulary.
        - dim_model (int): Dimensionality of the model.
        - num_heads (int): Number of attention heads in the multiheadattention models.
        - num_encoder_layers (int): Number of encoder layers in the Transformer.
        - num_decoder_layers (int): Number of decoder layers in the Transformer.
        - dropout_rate (float): Dropout rate to be applied in the model.
        """
        super().__init__()
        
        # Model Info
        self.model_type = "Transformer"
        self.dim_model = dim_model
        
        # Entire Layers
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_rate=dropout_rate, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer( # 여기도 직접 구현하면 더 좋을 것 같은데..
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_rate,
        )
        self.out = nn.Linear(dim_model, num_tokens)
    
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """
        왜 positional encoding 하기 전에 embedding 값에 sqrt(dim_model)을 곱하나요?
        """
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
    
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower trianular matrix : 하삼각행렬??
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_rate, max_len):
        super().__init__()
                
        self.dropout = nn.Dropout(dropout_rate)
        
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

