#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:33:31 2020

@author: wangjindong
"""


import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torch.nn.functional as F

# Define a 1-hidden layer neural network.
class model(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1,16,padding=1,kernel_size=3)
        self.dp = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,padding=1,kernel_size=3)
        self.conv3 = nn.Conv2d(32,64,padding=1,kernel_size=3)
        self.conv4 = nn.Conv2d(64,128,padding=1,kernel_size=3)
        self.conv5 = nn.Conv2d(128,192,padding=1,kernel_size=3)
        self.conv5t = nn.ConvTranspose2d(192,128,stride=2,kernel_size=3,padding=1,output_padding=1)
        self.conv44 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv4t = nn.ConvTranspose2d(128,64,stride=2,kernel_size=3,padding=1,output_padding=1)
        self.conv33 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.conv3t = nn.ConvTranspose2d(64,32,stride=2,kernel_size=3,padding=1,output_padding=1)
        self.conv22 = nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.conv2t = nn.ConvTranspose2d(32,16,stride=2,kernel_size=3,padding=1,output_padding=1)
        self.conv12 = nn.Conv2d(32,16,padding=1,kernel_size=3)
        self.conv11 = nn.Conv2d(16,1,padding=1,kernel_size=3)
        self.rl = nn.ReLU()
        self.th = nn.Tanh()
    def forward(self, x): 
        x1 = self.conv1(x)
        x1 = self.rl(x1)
        x2 = self.pool(x1)
        x2 = self.dp(x2)
        x2 = self.conv2(x2)
        x2 = self.rl(x2)
        x3 = self.pool(x2)
        x3 = self.dp(x3)
        x3 = self.conv3(x3)
        x3 = self.rl(x3)
        x4 = self.pool(x3)
        x4 = self.dp(x4)
        x4 = self.conv4(x4)
        x4 = self.rl(x4)
        x5 = self.pool(x4)
        x5 = self.dp(x5)
        x5 = self.conv5(x5)
        x5 = self.rl(x5)
        x44 = self.conv5t(x5)
        x44 = torch.cat([x4,x44],dim=1)
        x44 = self.dp(x44)
        x44 = self.conv44(x44)
        x44 = self.rl(x44)
        x33 = self.conv4t(x44)
        x33 = torch.cat([x3,x33],dim=1)
        x33 = self.dp(x33)
        x33 = self.conv33(x33)
        x33 = self.rl(x33)
        x22 = self.conv3t(x33)
        x22 = torch.cat([x2,x22],dim=1)
        x22 = self.dp(x22)
        x22 = self.conv22(x22)
        x22 = self.rl(x22)
        x11 = self.conv2t(x22)
        x11 = torch.cat([x1,x11],dim=1)
        x11 = self.dp(x11)
        x11 = self.conv12(x11)
        x11 = self.rl(x11)
        x11 = self.conv11(x11)
        x11 = self.th(x11)      
        return x11

def myloss(outputs, labels):
    loss = .5*torch.sum((outputs - labels)**2)
    return loss

