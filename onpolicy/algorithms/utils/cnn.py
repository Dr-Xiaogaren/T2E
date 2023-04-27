import torch.nn as nn
import torchvision.models as models
import torch
"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, args, input_channels, input_size):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU

        cov_block = [
                     nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                     nn.MaxPool2d(kernel_size=2, stride=2),
                     nn.BatchNorm2d(num_features=64),
                     nn.ReLU(),
                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                     nn.MaxPool2d(kernel_size=2, stride=2),
                     nn.BatchNorm2d(num_features=128),
                     nn.ReLU(),
                     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                     nn.MaxPool2d(kernel_size=2, stride=2),
                     nn.BatchNorm2d(num_features=256),
                     nn.ReLU()
                    ]
        self.Cov = nn.Sequential(*cov_block)
        test_input = torch.randn(1, input_channels, input_size, input_size)
        test_output = self.Cov(test_input)
        self.output_size = test_output.size(1)
        
    def forward(self, x):
        x = self.Cov(x)
        return torch.max(x.reshape(x.size(0),x.size(1),-1), -1)[0]

def main():
    from envs.mpe.environment import MultiAgentEnv, CatchingEnv
    from envs.mpe.scenarios import load
    from onpolicy.config import get_config
    parser = get_config()
    args = parser.parse_known_args()[0]
    args.num_agents = 4
    cnn_net = CNNBase(args,5,128)
    x = torch.randn(100,5,128,128)
    y = cnn_net(x)
    print(y.shape)

if __name__=="__main__":
   main()