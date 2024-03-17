import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_kernel():
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight


class VARM(nn.Module):
    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_ref = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_ref.append(_x)

        return torch.cat(x_ref, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        _imgs = self.get_dilated_neighbors(imgs)

        input_t = _imgs
        temp1 = torch.cat((input_t[:, :, :, 1:, :], input_t[:, :, :, -1, :].unsqueeze(3)), 3)
        temp2 = torch.cat((input_t[:, :, :, :, 1:], input_t[:, :, :, :, -1].unsqueeze(4)), 4)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)

        ref = -((_imgs_abs / (_imgs_std + 1e-8)) * 4) ** 2

        ref = ref.mean(dim=1, keepdim=True)

        temp = temp.mean(dim=1, keepdim=True)

        ref = F.softmax(ref, dim=2) - self.w2 * F.softmax(temp, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * ref).sum(2)

        return masks
