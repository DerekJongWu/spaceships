import torch
import torch.nn as nn

import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.L1 = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L2 = nn.Sequential(nn.Conv2d(32, 64, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L3 = nn.Sequential(nn.Conv2d(64, 128, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L4 = nn.Sequential(nn.Conv2d(128, 128, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.MaxPool2d(2, 2))
        self.L5 = nn.Sequential(nn.Conv2d(128, 256, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU())
        self.L6 = nn.Sequential(nn.Conv2d(256, 256, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU())
        self.L7 = nn.Sequential(nn.Conv2d(256, 16, 3, bias=True, padding=(1, 1)), nn.BatchNorm2d(16), nn.ReLU())

    def forward(self, x):
        return self.L7(self.L6(self.L5(self.L4(self.L3(self.L2(self.L1(x)))))))  # torch.Size([64, 16, 12, 12])


class Localizer(nn.Module):
    def __init__(self):
        super(Localizer, self).__init__()
        self.fc1 =  nn.Linear(16, 5)
        self.flat = nn.Flatten()
        self.an = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        return self.fc1(self.flat(self.an(x)))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16, 1)

        self.an = nn.AdaptiveAvgPool2d((1,1))
        self.sig = nn.Sigmoid() 

    def forward(self, x):
        return self.sig(self.fc1(self.flat(self.an(x))))
        


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input images 1 * 200 * 200

        self.convnet = ConvNet()
        self.localizer = Localizer()
        self.classifier = Classifier()

    def forward(self, x):
        x = self.convnet(x)
        classification = self.classifier(x)
        localization = self.localizer(x)
        
        return torch.cat((classification, localization), dim = 1) 

    def predict(self, x): 
        with torch.no_grad(): 
            pred = self.forward(x)
            pred = np.squeeze(pred)
            if pred[0] == 0:  
                print(1)
                return np.full(5, np.nan)
            else: 
                return pred[1:6].cpu().numpy()
