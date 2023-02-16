#Credits: https://github.com/mateuszbuda/brain-segmentation-pytorch

from collections import OrderedDict

import torch
import torch.nn as nn


class YNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32): 
        super(YNet, self).__init__()

        features = init_features
        
        ## CODER
        self.encoder1 = YNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = YNet._block(features * 8, features * 16, name="bottleneck")
        
        ## DECODER
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = YNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = YNet._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = YNet._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = YNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        
        ## Decoder Wounds
        
        self.upconv4w = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4w = YNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3w = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3w = YNet._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2w = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2w = YNet._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1w = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1w = YNet._block(features * 2, features, name="dec1")

        self.convw = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        ## Dec wound 3
        
        dec4w = self.upconv4(bottleneck)
        dec4w = torch.cat((dec4w, enc4), dim=1)
        dec4w = self.decoder4(dec4w)
        
        
        dec3w = self.upconv3(dec4w)
        dec3w = torch.cat((dec3w, enc3), dim=1)
        dec3w = self.decoder3(dec3w)
        
        dec2w = self.upconv2(dec3w)
        dec2w = torch.cat((dec2w, enc2), dim=1)
        dec2w = self.decoder2(dec2w)
        
        dec1w = self.upconv1(dec2w)
        dec1w = torch.cat((dec1w, enc1), dim=1)
        dec1w = self.decoder1(dec1w)
        
        return torch.sigmoid(self.conv(dec1)), torch.sigmoid(self.convw(dec1w))

    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class YNetGN(YNet):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, to_train = True):
        super(YNetGN, self).__init__(in_channels, out_channels, init_features)

        features = init_features
            
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        
        self.loss1 = DiceLoss()
        self.loss2 = DiceLoss()
        
        self.to_train = to_train


    def forward(self, x, y=None):
        out1, out2 =super().forward(x)
        
        if self.to_train:
            loss1 = self.loss1(out1, y[0])
            loss2 = self.loss2(out2, y[1])
            
            return torch.stack([loss1, loss2])
        else:
            return [out1, out2]
            
