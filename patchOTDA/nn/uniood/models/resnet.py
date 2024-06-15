import torch.nn as nn

class ResBase(nn.Module):
    def __init__(self, option='resnet50', pretrained=True):
        super(ResBase, self).__init__()
        raise NotImplementedError
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.output_dim)
        return x

