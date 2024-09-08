import torch
import torch.nn as nn
import torch.nn.functional as F

architecture_config = [
    #(kernel size, number of filters, stride, padding)
    (7, 64, 2, 3),
    "M", #max pool
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # to be repeated 4 times, hence the 4
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],# to be repeated 2 times, hence the 2
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leakyrelu(self.batchnorm(self.conv(x)))
        return x

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # x = x.view(x.shape[0], -1) can be used and only x can be passed
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNN(
                    in_channels=in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                )]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            elif type(x) == list:
                for _ in range(x[2]):
                    layers += [CNN(
                        in_channels=in_channels, out_channels=x[0][1], kernel_size=x[0][0], stride=x[0][2], padding=x[0][3]
                    )]
                    layers += [CNN(
                        in_channels=x[0][1], out_channels=x[1][1], kernel_size=x[1][0], stride=x[1][2], padding=x[1][3]
                    )]

                    in_channels = x[1][1]
        
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(), # not really needed
            nn.Linear(1024*S*S, 496), # 4096 in the original paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C + B*5)),
        )

# run the below ufunction to check for the implementation of the model  
def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

