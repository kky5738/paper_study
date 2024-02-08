import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x):
        out = self.encoder(x)
        return out
    
    def decode(self, x):
        out = self.decoder(x)
        return out
    
    def forward(self, x, z):
        c = self.encode(x)
        y = self.encode(z, c) # z는 target인가?
        return y