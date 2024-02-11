import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import math

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out
    
    def decode(self, x):
        out = self.decoder(x)
        return out
    
    def forward(self, src, tgt, src_mask):
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out) 
        return y
    
class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
        
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        
    def forward(self, src, src_mask):
        out = src
        out = self.self_attention(query=out, key=out, value=out, mask=src_mask)
        out = self.position_ff(out)
        return out

def calculate_attention(query, key, value, mask):
    # query, key, value: (n_batch, seq_len, dim_key)
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1))
    attention_score = attention_score / math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9) # masked_fill 어떻게 동작하는지 찾아보기
        # 아마도 masked_fill은 pad_token에 해당하는 0을 찾고, 0 대신 -1e9(음의 무한대) 값으로 바꾸는 듯
    attention_prob = F.softmax(attention_score, dim=-1) # torch에서 softmax의 dim 옵션 뭔지 찾아보기
    out = torch.matmul(attention_prob, value)
    return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc, calculate_attention):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc) # kfc
        self.v_fc = copy.deepcopy(qkv_fc)
        # 근데 deepcopy로 복사해도 어차피 초기 weight는 같을텐데 학습 과정에서 달라지려나?? ㄷ일단 shallow copy는 당연히 쓰면 x. 새로 layer 만드는 거랑 무슨 차이가 있나요
        # 정리하면 deepcopy로 layer를 복사하는 거랑 그냥 새로 layer 만드는 거랑 무슨 차이가 있나요
        self.calculate_attention = calculate_attention
        self.out_fc = out_fc
    
    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, dim_embed)
        n_batch = query.size(0)
        
        def transform(x, fc): # linear layer를 종종 어떤 논문에서는 affine transform? 으로 표현하기도 함.
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            out = out.transpose(1, 2)
            return out
        
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)
        
        out = self.calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2) # transpose가 종종 등장함. 아마 fc나 attention 거칠 때 내부에서 행렬 계산을 위해 transpose가 있어서 그러는 듯 # 헷갈리면 직접 dim 계산 해보기
        out = out.contiguous().view(n_batch, -1, self.d_model) # contiguous()가 무슨 함수인지 torch 문서 찾아보기
        out = self.out_fc(out)
        return out
        