from torch import nn
#for a backbone, we will use a simple feedforward neural network
class FFNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FFNet, self).__init__()

        self.output_dim = out_dim
        self.input_dim = in_dim
        #simple 3 layer feedforward neural network
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):
        output = self.fc(x)
        return output
