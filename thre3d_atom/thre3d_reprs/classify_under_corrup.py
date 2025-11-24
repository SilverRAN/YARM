from transformers import logging
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
import random
from .corruptions import random_corruption

class ClassificationLoss(nn.Module):
    def __init__(self,
                 device,
                 proxy_model = 'resnet18',
                ):
        super().__init__()
        self.proxy_model = proxy_model
        if self.proxy_model == 'resnet18':
            self.classifier = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        elif self.proxy_model == 'vgg16':
            self.classifier = vgg16(weights=VGG16_Weights.DEFAULT).to(device)
        else:
            raise ValueError(f'Proxy model {self.proxy_model} not supported.')
        self.classifier.eval()

    def training_step(self, output, image_height, image_width, mask=None, label=None):
        loss = 0

        out_imgs = torch.reshape(output, (-1, image_height, image_width, 3))
        out_imgs = out_imgs.permute((0, 3, 1, 2))
        if mask is not None:
            B, _, H, W = out_imgs.shape
            bg = torch.concat([torch.ones((B, 1, H, W)) * random.random(), torch.ones((B, 1, H, W)) * random.random(), torch.ones((B, 1, H, W)) * random.random()], dim=1).to(device=out_imgs.device)
            out_imgs = mask * out_imgs + (1 - mask) * bg
        # get classification logits
        out_imgs = random_corruption(out_imgs)
        '''Add classification model here.'''
        if self.proxy_model == 'resnet18':
            preprocess = ResNet18_Weights.DEFAULT.transforms(antialias=True)
        elif self.proxy_model == 'vgg16':
            preprocess = VGG16_Weights.DEFAULT.transforms(antialias=True)
        noisy_pred = self.classifier(preprocess(out_imgs))
        loss = F.cross_entropy(noisy_pred, torch.tensor([label], device=out_imgs.device).long())

        return loss