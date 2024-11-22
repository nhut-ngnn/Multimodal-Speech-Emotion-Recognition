import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, text_embed_dim=480, audio_embed_dim=480, num_classes=4, dropout_rate=0.3):
        super(FlexibleMMSER, self).__init__()
        concat_embed_dim = text_embed_dim + audio_embed_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_embed_dim, 960),
            nn.BatchNorm1d(960),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(960, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text_embed, audio_embed):
        concat_embed = torch.cat((text_embed, audio_embed), dim=1)
        projected_embed = self.projection(concat_embed)
        x = self.dropout(projected_embed)
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        y_logits = self.linear3(x)
        y_softmax = self.softmax(y_logits)
        return y_logits, y_softmax
