import torch.nn as nn

class CNNLayerNorm(nn.Module):
    """Layer Normalization built for CNNs input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x): # x(batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # x(batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # x(batch, channel, feature, time)

class ResidualCNN(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass
    
class BidirectionalGRU(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass 

class SpeechRecognitionModel(nn.Module):
    def __init__(self, rnn_dim, n_class, dropout=0.1):
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim), #birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x