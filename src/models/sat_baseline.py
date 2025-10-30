import torch, torch.nn as nn
import torchvision.models as models
from vocab import PAD, SOS, EOS

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=512, freeze_backbone=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(m.children())[:-1]   # remove fc => (B,2048,1,1)
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)     # (B,2048)
        feat = self.fc(feat)                   # (B,embed)
        return feat

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=1, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # features: (B,embed)  captions: (B,T)
        emb = self.embed(captions)             # (B,T,E)
        # prepend feature as first step (como 'context vector')
        feat = features.unsqueeze(1)           # (B,1,E)
        inp = torch.cat([feat, emb[:, :-1, :]], dim=1)  # teacher forcing
        out, _ = self.lstm(inp)
        logits = self.fc(out)                  # (B,T,V)
        return logits

class SATBaseline(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim=512, freeze_backbone=True)
        self.decoder = DecoderRNN(vocab_size, embed_dim=512, hidden_dim=512, pad_idx=pad_idx)

    def forward(self, images, captions):
        feats = self.encoder(images)
        logits = self.decoder(feats, captions)
        return logits
