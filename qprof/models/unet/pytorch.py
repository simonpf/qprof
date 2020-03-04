import torch
from torch import nn

class Block(nn.Sequential):
    def __init__(self,
                 in_features,
                 out_features,
                 depth=2,
                 activation=nn.ReLU(),
                 no_activation = False,
                 mode="none"):

        # First layer, increase number of features.
        if depth == 1:
            if no_activation:
                modules = [nn.ConstantPad2d(1, 0.0),
                           nn.Conv2d(in_features, out_features, 3)]
            else:
                modules = [nn.ConstantPad2d(1, 0.0),
                           nn.Conv2d(in_features, out_features, 3),
                           nn.BatchNorm2d(out_features),
                           activation]
        else:
            modules = [nn.ConstantPad2d(1, 0.0),
                       nn.Conv2d(in_features, out_features, 3),
                       nn.BatchNorm2d(out_features),
                       activation]
            if no_activation:
                modules += [nn.ConstantPad2d(1, 0.0),
                            nn.Conv2d(out_features, out_features, 3),
                            nn.BatchNorm2d(out_features),
                            activation] * max(depth - 2, 0)
                modules += [nn.ConstantPad2d(1, 0.0),
                            nn.Conv2d(out_features, out_features, 3),
                            nn.BatchNorm2d(out_features)]
            else:
                modules += [nn.ConstantPad2d(1, 0.0),
                            nn.Conv2d(out_features, out_features, 3),
                            nn.BatchNorm2d(out_features),
                            activation] * max(depth - 1, 0)

        super().__init__(*modules)

class DownSampler(nn.Sequential):
    def __init__(self):
        modules = [nn.MaxPool2d(2)]
        super().__init__(*modules)

class UpSampler(nn.Sequential):
    def __init__(self,
                 features_in,
                 features_out):
        modules = [nn.ConvTranspose2d(features_in,
                                      features_out,
                                      3,
                                      padding=1,
                                      output_padding=1,
                                      stride=2)]
        super().__init__(*modules)

class UNet(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 n_features=32,
                 n_levels=4):

        super().__init__()

        # Down-sampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        features_in = input_features
        features_out = n_features
        for i in range(n_levels - 1):
            self.down_blocks.append(Block(features_in, features_out))
            self.down_samplers.append(DownSampler())
            features_in = features_out
            features_out = features_out * 2

        self.center_block = Block(features_in, features_out)

        self.up_blocks = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        features_in = features_out
        for i in range(n_levels - 1):
            self.up_samplers.append(UpSampler(features_in, features_in // 2))
            self.up_blocks.append(Block(features_in, features_in // 2))
            features_in = features_in // 2

        self.head = nn.Conv2d(features_in, output_features, 3)
        self.head = nn.Sequential(nn.ConstantPad2d(1, 0.0),
                                  nn.Conv2d(features_in, output_features, 3))

    def forward(self, x):

        features = []
        for (b, s) in zip(self.down_blocks, self.down_samplers):
            x = b(x)
            features.append(x)
            x = s(x)

        x = self.center_block(x)

        for (b, u, f) in zip(self.up_blocks, self.up_samplers, features[::-1]):
            x = u(x)
            x = torch.cat([x, f], 1)
            x = b(x)

        self.features = features

        return self.head(x)
