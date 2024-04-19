import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from collections import Counter


class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)  # (4, 2048, 1, 1)
        features = Variable(features.data) 
        features = features.view(features.size(0), -1) # (4, 2048)
        features = self.linear(features)    # (4, 1024)
        features = self.bn(features)        # (4, 1024)
        return features
    

class EncoderYOLO(nn.Module):

    def __init__(self, target_size):
        super(EncoderYOLO, self).__init__()

        self.yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        for param in self.yolo.model.parameters():
            param.requires_grad = False

        # register activation to store 3 feature maps
        self.activation = {}
        def get_input():
            def hook(model, input, output):
                data = input[0]
                self.activation["large"] = data[0].detach()
                self.activation["medium"] = data[1].detach()
                self.activation["small"] = data[2].detach()
            return hook
        self.yolo.model.model.model[24].register_forward_hook(get_input())

        self.linear = nn.Linear(3 * 8 * 8 * 85, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)

        return

    def forward(self, images):
        _ = self.yolo(images)

        act = self.activation["small"] # (B, 3, 8, 8, 85)
        act = torch.clone(act)

        B, _, _, _, _ = act.shape

        act = act.reshape(B, -1)
        act = self.linear(act)
        act = self.bn(act)
        return act

if __name__ == "__main__":

    B = 4
    C = 3
    H = 256
    W = 256
    target_size = 1024

    imgs = torch.randn(B, C, H, W)

    encoder = EncoderYOLO(target_size)
    out = encoder(imgs)
    print(out.shape)