import torch.nn as nn

class CNNLayerNorm(nn.Module):
    """Layer Normalization built for CNNs input"""
    def __init__(self, n_feats):
        pass

    def forward(self, x):
        pass

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
    def __init__(self):
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim), #birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x