import torch.nn as nn
import torch.nn.Functional as F

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
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel , stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x #(batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual 
        return x #(batch, channel, feature, time)
    
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