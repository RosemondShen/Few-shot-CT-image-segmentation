
import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
import torch.nn.functional as F

class FewShotSegmentorDoubleSDnet(nn.Module):

    def __init__(self, params):
        super(FewShotSegmentorDoubleSDnet, self).__init__()
        '''
        self.conditioner = SDnetConditioner(params)
        self.segmentor = SDnetSegmentor(params)
        '''

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode1 = sm.SDnetDecoderBlock(params)

        params['num_channels'] = 128
        # self.classifier = sm.ClassifierBlock(params)
        # self.classifier = nn.Conv2d(128, 1, 1, 1)
        self.classifier = nn.Conv2d(64, 1, 1, 1)

        self.net1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2), stride=1)
        self.net2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()

    def forward(self, input2):

        que_e1, que_ind1 = self.encode1(input2)
        que_e2, que_ind2 = self.encode2(que_e1)
        que_e3, que_ind3 = self.encode3(que_e2)
        que_e4, que_ind4 = self.encode4(que_e3)
        que_bn = self.bottleneck(que_e4)
        que_d4 = self.decode4(que_bn, que_ind4)
        que_d4 = torch.cat([que_e3, que_d4], dim=1)
        que_d4 = self.net1(que_d4)
        que_d3 = self.decode3(que_d4, que_ind3)
        que_d3 = torch.cat([que_e2, que_d3], dim=1)
        que_d3 = self.net1(que_d3)
        que_d2 = self.decode2(que_d3, que_ind2)
        que_d2 = torch.cat([que_e1, que_d2], dim=1)
        que_d2 = self.net1(que_d2)
        que_d1 = self.decode1(que_d2, que_ind1)

        que_d1 = self.net2(que_d1)

        pred = self.classifier(que_d1)

        # weights = self.conditioner(input1)
        # segment = self.segmentor(input2, weights)
        # return segment

        return pred
