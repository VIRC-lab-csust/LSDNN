import copy
import random

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck, resnet50
from config import cfg
from models.resnet_ibn_a import resnet50_ibn_a
from models.senet import SENet, SEResNeXtBottleneck
from utils.init import weights_init_kaiming


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_zeros(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 1
            x = x * mask
        return x


class BagReID_SE_RESNEXT(nn.Module):

    def __init__(self, num_classes=0, width_ratio=0.5, height_ratio=0.5):
        super(BagReID_SE_RESNEXT, self).__init__()

        self.backbone = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=1)

        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_bn = nn.BatchNorm1d(1024)
        self.global_softmax = nn.Linear(1024, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        self.global_reduction.apply(weights_init_kaiming)

        # part branch
        self.part = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_drop = BatchDrop(height_ratio, width_ratio)
        self.part_reduction = nn.Sequential(
            nn.Linear(2048, 1024, True),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )

        self.part_reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(1024)
        self.part_softmax = nn.Linear(1024, num_classes)
        self.part_softmax.apply(weights_init_kaiming)

    def forward(self, x):

        x = self.backbone(x)
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).view(glob.size(0), -1)
        triplet_features.append(global_triplet_feature)
        
        #part1 branch
        x = self.part(x)
        z0_p3 = self.part_maxpool(x[:, :, 0:1, :]).view(x.size(0), -1)
        z1_p3 = self.part_maxpool(x[:, :, 1:2, :]).view(x.size(0), -1)
        z2_p3 = self.part_maxpool(x[:, :, 2:3, :]).view(x.size(0), -1)
        
        f0_p3 = self.part_reduction(z0_p3)
        f1_p3 = self.part_reduction(z1_p3)
        f2_p3 = self.part_reduction(z2_p3)
  
        # part2 branch
        x = self.batch_drop(x)
        triplet_feature = self.part_maxpool(x).view(x.size(0), -1)
        feature = self.part_reduction(triplet_feature)
        triplet_features.append(feature)
    
       
        if self.training:
            global_softmax_class = self.global_softmax(self.global_bn(global_triplet_feature))
            softmax_features.append(global_softmax_class)

            softmax_feature = self.part_softmax(self.part_bn(feature))
            softmax_feature1 = self.part_softmax(self.part_bn(f0_p3))
            softmax_feature2 = self.part_softmax(self.part_bn(f1_p3))
            softmax_feature3 = self.part_softmax(self.part_bn(f2_p3))
            softmax_features.append(softmax_feature)  
            softmax_features.append(softmax_feature1)
            softmax_features.append(softmax_feature2)
            softmax_features.append(softmax_feature3)
            return triplet_features, softmax_features
        else:
            return torch.cat(triplet_features, dim=1)


class BagReID_RESNET(nn.Module):

    def __init__(self, num_classes=0, width_ratio=0.5, height_ratio=0.5):
        super(BagReID_RESNET, self).__init__()

        resnet = resnet50(pretrained=True)
        layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        layer4.load_state_dict(resnet.layer4.state_dict())

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            layer4
        )

        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_bn = nn.BatchNorm1d(cfg.MODEL.GLOBAL_FEATS, affine=False)
        self.global_softmax = nn.Linear(cfg.MODEL.GLOBAL_FEATS, num_classes, bias=False)
        self.global_softmax.apply(weights_init_kaiming)
        # self.global_reduction = nn.Sequential(
        #     nn.Conv2d(2048, cfg.MODEL.GLOBAL_FEATS, 1),
        #     nn.BatchNorm2d(cfg.MODEL.GLOBAL_FEATS),
        #     nn.ReLU(True)
        # )
        # self.global_reduction.apply(weights_init_kaiming)

        # part branch
        self.part = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_drop = BatchDrop(height_ratio, width_ratio)
        self.part_reduction = nn.Sequential(
            nn.Linear(2048, cfg.MODEL.PART_FEATS, True),
            nn.BatchNorm1d(cfg.MODEL.PART_FEATS),
            nn.ReLU(True)
        )

        self.part_reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(cfg.MODEL.PART_FEATS, affine=False)
        self.part_softmax = nn.Linear(cfg.MODEL.PART_FEATS, num_classes, bias=False)
        self.part_softmax.apply(weights_init_kaiming)

    def forward(self, x):

        x = self.backbone(x)
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = glob.view(glob.size(0), -1)
        # global_triplet_feature = self.global_reduction(glob).view(glob.size(0), -1)
        triplet_features.append(global_triplet_feature)

        # part branch
        x = self.part(x)
        x = self.batch_drop(x)
        part_triplet_feature = self.part_maxpool(x).view(x.size(0), -1)
        part_triplet_feature = self.part_reduction(part_triplet_feature)
        triplet_features.append(part_triplet_feature)

        if self.training:
            global_softmax_class = self.global_softmax(self.global_bn(global_triplet_feature))
            softmax_features.append(global_softmax_class)
            softmax_feature = self.part_softmax(self.part_bn(part_triplet_feature))
            softmax_features.append(softmax_feature)
            return triplet_features, softmax_features
        else:
            return torch.cat(triplet_features, 1)


class BagReID_IBN(nn.Module):

    def __init__(self, num_classes_cls=0, num_classes_mate=0, width_ratio=0.7, height_ratio=0.7):
        super(BagReID_IBN, self).__init__()

        self.backbone = resnet50_ibn_a(last_stride=1, pretrained=True)
        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_bn = nn.BatchNorm1d(2048)
        # ID损失 500类
        self.global_softmax_cls = nn.Linear(2048, num_classes_cls)
        # mate损失 4类
        self.global_softmax_mate = nn.Linear(2048, num_classes_mate)
        self.global_softmax_cls.apply(weights_init_kaiming)
        self.global_softmax_mate.apply(weights_init_kaiming)
        self.global_reduction = nn.Sequential(
            nn.Conv2d(2048, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )
        self.global_reduction.apply(weights_init_kaiming)

        # part branch
        self.part = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_drop = BatchDrop(height_ratio, width_ratio)
        self.part_reduction = nn.Sequential(
            nn.Linear(2048, 2048, True),
            nn.BatchNorm1d(2048),
            nn.ReLU(True)
        )

        self.part_reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(2048)
        self.part_softmax_cls = nn.Linear(2048, num_classes_cls)
        self.part_softmax_mate = nn.Linear(2048, num_classes_mate)
        self.part_softmax_cls.apply(weights_init_kaiming)
        self.part_softmax_mate.apply(weights_init_kaiming)

    def forward(self, x):

        # 输出 (batch, 2048, 24, 8)
        x = self.backbone(x)
        predict = []
        triplet_features = []
        softmax_features_cls = []
        softmax_features_mate = []

        # global branch
        # 输出 (batch, 2048, 1, 1)
        glob = self.global_avgpool(x)
        glob_softmax = glob.squeeze()
        # 输出 (batch, 2048)
        global_triplet_feature = self.global_reduction(glob).view(glob.size(0), -1)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        # 输出 (batch, 2048, h, w)
        x = self.part(x)

        #part1 branch
        # 输出 (batch, 2048)
        part1_triplet_feature = self.part_maxpool(x).view(x.size(0), -1)
        # 输出 (batch, 2048)
        part1_feature = self.part_reduction(part1_triplet_feature)
        triplet_features.append(part1_feature)
        predict.append(part1_feature)

        # 输出 (batch, 2048)
        z0_p3 = self.part_maxpool(x[:, :, 0:1, :]).view(x.size(0), -1)
        # 输出 (batch, 2048)
        z1_p3 = self.part_maxpool(x[:, :, 1:2, :]).view(x.size(0), -1)
        # 输出 (batch, 2048)
        z2_p3 = self.part_maxpool(x[:, :, 2:3, :]).view(x.size(0), -1)
        
        # 输出 (batch, 2048)
        #f0_p3 = self.part_reduction(z0_p3)
        # 输出 (batch, 2048)
        f1_p3 = self.part_reduction(z1_p3)
        # 输出 (batch, 2048)
        #f2_p3 = self.part_reduction(z2_p3)
  
        # part2 branch
        # 输出 (batch, 2048, h, w)
        x2 = self.batch_drop(x)
        # 输出 (batch, 2048)
        part2_triplet_feature = self.part_maxpool(x2).view(x2.size(0), -1)
        # 输出 (batch, 2048)
        part2_feature = self.part_reduction(part2_triplet_feature)
        triplet_features.append(part2_feature)
        predict.append(part2_feature)
    
       
        if self.training:
            # ID损失, 输出 (batch, num_classes_cls)
            global_softmax_class_cls = self.global_softmax_cls(self.global_bn(global_triplet_feature))
            # mate损失, 输出 (batch, num_classes_mate)
            global_softmax_class_mate = self.global_softmax_mate(self.global_bn(global_triplet_feature))
            softmax_features_cls.append(global_softmax_class_cls)
            softmax_features_mate.append(global_softmax_class_mate)
            
            # ID损失
            softmax_feature0_cls = self.part_softmax_cls(self.part_bn(part2_triplet_feature))

            softmax_feature_cls = self.part_softmax_cls(self.part_bn(part1_triplet_feature))
            #softmax_feature1_cls = self.part_softmax_cls(self.part_bn(f0_p3))
            softmax_feature2_cls = self.part_softmax_cls(self.part_bn(f1_p3))
            #softmax_feature3_cls = self.part_softmax_cls(self.part_bn(f2_p3))

            softmax_features_cls.append(softmax_feature0_cls)
            softmax_features_cls.append(softmax_feature_cls)  
            #softmax_features_cls.append(softmax_feature1_cls)
            softmax_features_cls.append(softmax_feature2_cls)
            #softmax_features_cls.append(softmax_feature3_cls)
            
            # mate损失
            softmax_feature0_mate = self.part_softmax_mate(self.part_bn(part2_triplet_feature))

            softmax_feature_mate = self.part_softmax_mate(self.part_bn(part1_triplet_feature))
            #softmax_feature1_mate = self.part_softmax_mate(self.part_bn(f0_p3))
            softmax_feature2_mate = self.part_softmax_mate(self.part_bn(f1_p3))
            #softmax_feature3_mate = self.part_softmax_mate(self.part_bn(f2_p3))

            softmax_features_mate.append(softmax_feature0_mate)
            softmax_features_mate.append(softmax_feature_mate)  
            #softmax_features_mate.append(softmax_feature1_mate)
            softmax_features_mate.append(softmax_feature2_mate)
            #softmax_features_mate.append(softmax_feature3_mate)
            
            return triplet_features, softmax_features_cls, softmax_features_mate
        else:
            return torch.cat(predict, dim=1)


class BagReID(nn.Module):
    def __init__(self, num_classes=0, width_ratio=0.5, height_ratio=0.5):
        super(BagReID, self).__init__()
        self.part1 = BagReID_SE_RESNEXT(num_classes, width_ratio, height_ratio)
        self.part2 = BagReID_RESNET(num_classes, width_ratio, height_ratio)
        self.part3 = BagReID_IBN(num_classes, width_ratio, height_ratio)
        self.part1.load_state_dict(
            torch.load('./snapshot/resnext50.pth.tar', map_location='cpu')['state_dict'])
        self.part2.load_state_dict(
            torch.load('./snapshot/resnet50.pth.tar', map_location='cpu')['state_dict'])
        self.part3.load_state_dict(
            torch.load('./snapshot/resnet50_ibn_a.pth.tar', map_location='cpu')['state_dict'])

    def forward(self, x):
        part1 = self.part1(x)
        part2 = self.part2(x)
        part3 = self.part3(x)
        return torch.cat((part1, part2, part3), dim=1)
        # return part2
