import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMMSER(nn.Module):
    def __init__(self, num_classes=4):
        super(FlexibleMMSER, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(.2)
        self.linear = nn.Linear(768*2, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, text_embed, audio_embed):
        concat_embed = torch.cat((text_embed, audio_embed), dim=1)
        # x = self.dropout(text_embed)
        x = self.linear(concat_embed)
        x = self.linear1(x)
        x = self.linear2(x)
        y_logits = self.linear3(x)
        y_softmax = self.softmax(y_logits)
        return y_logits, y_softmax