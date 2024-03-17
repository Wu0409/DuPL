import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder


class network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, aux_layer=None, add_mlp=False):
        super().__init__()
        self.num_classes = num_classes
        self.add_mlp = add_mlp

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.in_channels = \
            [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4

        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.LargeFOV(
            in_planes=self.in_channels[-1], out_planes=self.num_classes
        )

        self.classifier = nn.Conv2d(
            in_channels=self.in_channels[-1], out_channels=self.num_classes - 1, kernel_size=1, bias=False
        )

        self.aux_classifier = nn.Conv2d(
            in_channels=self.in_channels[-1], out_channels=self.num_classes - 1, kernel_size=1, bias=False
        )

        # v6
        if self.add_mlp:
            self.mlp_layer = nn.Sequential(
                nn.Conv2d(self.in_channels[-1], self.in_channels[-1], 1, bias=False),
                # nn.BatchNorm2d(self.in_channels[-1]),
                # nn.ReLU(True),
                # nn.Dropout2d(p=0.5)
            )

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.mlp_layer.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, cam_only=False, val=False, cam_with_grad=False):

        cls_token, _x, x_aux = self.encoder.forward_features(x)

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w)
        _x_aux = self.to_2D(x_aux, h, w)

        if self.add_mlp:
            _x4 = self.mlp_layer(_x4)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
            return cam_aux, cam

        seg = self.decoder(_x4)

        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        cls_aux = cls_aux.view(-1, self.num_classes - 1)

        if val:
            return cls_x4, seg, _x4, cls_aux

        if cam_with_grad:
            cam_grad = F.conv2d(_x4, self.classifier.weight.detach())
            cam_grad = cam_grad + F.adaptive_max_pool2d(-cam_grad, (1, 1))
            cam_grad = cam_grad / F.adaptive_max_pool2d(cam_grad, (1, 1)) + 1e-5
            return cls_x4, seg, _x4, cls_aux, cam_grad

        return cls_x4, seg, _x4, cls_aux


class siamese_network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, aux_layer=None):
        super().__init__()
        self.branch1 = network(
            backbone, num_classes=num_classes, pretrained=pretrained, aux_layer=aux_layer, add_mlp=False
        )
        self.branch2 = network(
            backbone, num_classes=num_classes, pretrained=pretrained, aux_layer=aux_layer, add_mlp=False
        )

    def get_param_groups(self):
        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;
        # encoder
        for name, param in list(self.branch1.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for name, param in list(self.branch2.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # classifier
        param_groups[2].append(self.branch1.classifier.weight)
        param_groups[2].append(self.branch1.aux_classifier.weight)
        param_groups[2].append(self.branch2.classifier.weight)
        param_groups[2].append(self.branch2.aux_classifier.weight)

        # proj
        if self.branch1.add_mlp:
            for param in list(self.branch1.mlp_layer.parameters()):
                param_groups[2].append(param)
        if self.branch2.add_mlp:
            for param in list(self.branch2.mlp_layer.parameters()):
                param_groups[2].append(param)

        # decoder
        for param in list(self.branch1.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.branch2.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x, val=False, cam_only=False, cam_with_grad=False, branch=None, need_sp=False):
        res = {}
        b, c, h, w = x.shape

        if val:
            if branch is None:  # both (default)
                res["branch1"] = self.branch1(x)
                res["branch2"] = self.branch2(x)
                return res
            elif branch == 1:  # branch1
                return self.branch1(x)
            else:  # branch 2
                return self.branch2(x)

        if cam_only:
            if branch is None:
                cam_aux_1, cam_1 = self.branch1(x, cam_only=cam_only)
                cam_aux_2, cam_2 = self.branch2(x, cam_only=cam_only)
                return cam_aux_1, cam_1, cam_aux_2, cam_2
            elif branch == 1:
                return self.branch1(x, cam_only=cam_only)
            else:
                return self.branch2(x, cam_only=cam_only)

        if cam_with_grad:
            if branch is None:  # both
                res["branch1"] = self.branch1(x, cam_with_grad=True)
                res["branch2"] = self.branch2(x, cam_with_grad=True)
                return res
            elif branch == 1:  # branch1
                return self.branch1(x, cam_with_grad=True)
            else:  # branch 2
                return self.branch2(x, cam_with_grad=True)

        if branch is None:  # both
            res["branch1"] = self.branch1(x)
            res["branch2"] = self.branch2(x)

            if need_sp:
                x, x_aug = x.chunk(2)
                x_aug = F.interpolate(x_aug, scale_factor=0.75, mode="bilinear", align_corners=False)

                res["branch1"] = self.branch1(x)
                res["branch2"] = self.branch2(x)

                _, seg_1_aug, _, _ = self.branch1(x_aug)
                _, seg_2_aug, _, _ = self.branch2(x_aug)

                res["branch1_aug"] = seg_1_aug
                res["branch2_aug"] = seg_2_aug

            return res

        elif branch == 1:  # branch1
            return self.branch1(x)

        else:  # branch 2
            return self.branch2(x)

