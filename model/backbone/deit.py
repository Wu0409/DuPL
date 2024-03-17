# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from email.policy import strict
import torch
import torch.nn as nn
from functools import partial

from .vit import VisionTransformer, VisionTransformer_with_aff, _cfg, VisionTransformer_mct, VisionTransformer_memo, \
    VisionTransformer_mask, VisionTransformer_mask_v1, VisionTransformer_with_attn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        patch_embed = x

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = torch.cat((dist_token, cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights[:, :, 1:, 1:])

        x = self.norm(x)
        return x[:, 1], x[:, 2:], attn_weights, patch_embed

    def forward(self, x):
        x, attn_weights = self.forward_features(x)
        # x = self.head(x)
        # x_dist = self.head_dist(x_dist)
        if self.training:
            return x
        else:
            # during inference, return the average of both classifier predictions
            return x, attn_weights


# @register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )["model"]
        model.load_state_dict(checkpoint)
    return model


# @register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )["model"]
        model.load_state_dict(checkpoint)
    return model


def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]
        model.load_state_dict(checkpoint)
    return model


def deit_base_patch16_224_with_attn(pretrained=False, **kwargs):
    model = VisionTransformer_with_attn(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]
        model.load_state_dict(checkpoint)
    return model


def deit_base_patch16_224_mask(pretrained=False, **kwargs):
    model = VisionTransformer_mask(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]
        # model.load_state_dict(checkpoint)

        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

        cls_token = pretrained_dict['cls_token'].repeat(1, 1, 1)
        pos_embed_cls = pretrained_dict['pos_embed'][:, 0, :].repeat(1, 1, 1)
        pos_embed_patch = pretrained_dict['pos_embed'][:, 1:, :]
        pos_embed = torch.cat((pos_embed_cls, pos_embed_patch), dim=1)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}

        pretrained_dict['cls_token'] = cls_token
        pretrained_dict['pos_embed'] = pos_embed

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# for v7
def deit_base_patch16_224_mask_v1(pretrained=False, **kwargs):
    model = VisionTransformer_mask_v1(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]
        # model.load_state_dict(checkpoint)

        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

        cls_token = pretrained_dict['cls_token'].repeat(1, 1, 1)
        pos_embed_cls = pretrained_dict['pos_embed'][:, 0, :].repeat(1, 1, 1)
        pos_embed_patch = pretrained_dict['pos_embed'][:, 1:, :]
        pos_embed = torch.cat((pos_embed_cls, pos_embed_patch), dim=1)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}

        pretrained_dict['cls_token'] = cls_token
        pretrained_dict['pos_embed'] = pos_embed

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def deit_base_patch16_224_dino(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    state_dict = torch.load(
        '/media/store/wyc/exps/WSSS_2023/pretrained/dino_vitbase16_pretrain.pth', map_location="cpu"
    )

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    return model


def deit_base_patch16_224_memo(pretrained=False, **kwargs):
    model = VisionTransformer_memo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]
        # model.load_state_dict(checkpoint)

        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

        cls_token = pretrained_dict['cls_token'].repeat(1, 1, 1)
        pos_embed_cls = pretrained_dict['pos_embed'][:, 0, :].repeat(1, 1, 1)
        pos_embed_patch = pretrained_dict['pos_embed'][:, 1:, :]
        pos_embed = torch.cat((pos_embed_cls, pos_embed_patch), dim=1)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}

        pretrained_dict['cls_token'] = cls_token
        pretrained_dict['pos_embed'] = pos_embed

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def deit_base_patch16_224_mct(pretrained=False, **kwargs):
    model = VisionTransformer_mct(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]

        # model.load_state_dict(checkpoint)

        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

        cls_token = pretrained_dict['cls_token'].repeat(1, 20, 1)
        pos_embed_cls = pretrained_dict['pos_embed'][:, 0, :].repeat(1, 20, 1)
        pos_embed_patch = pretrained_dict['pos_embed'][:, 1:, :]
        pos_embed = torch.cat((pos_embed_cls, pos_embed_patch), dim=1)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}

        pretrained_dict['cls_token'] = cls_token
        pretrained_dict['pos_embed'] = pos_embed

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def deit_base_patch16_224_with_aff(pretrained=False, **kwargs):
    model = VisionTransformer_with_aff(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/eit_base_patch16_224-b5f2ef4d.pth",
            model_dir='./pretrained',
            map_location="cpu", check_hash=True
        )["model"]
        model.load_state_dict(checkpoint)
    return model


# @register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )["model"]
        checkpoint.pop("head_dist.weight")
        checkpoint.pop("head_dist.bias")
        model.load_state_dict(checkpoint)
    return model


# @register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )["model"]
        checkpoint.pop("head_dist.weight")
        checkpoint.pop("head_dist.bias")
        model.load_state_dict(checkpoint)
    return model


# @register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )["model"]
        checkpoint.pop("head_dist.weight")
        checkpoint.pop("head_dist.bias")
        model.load_state_dict(checkpoint)
    return model


# @register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            # url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
            map_location="cpu", check_hash=True
        )["model"]
        model.load_state_dict(checkpoint)
    return model


# @register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )["model"]
        checkpoint.pop("head_dist.weight")
        checkpoint.pop("head_dist.bias")
        ## swap pos embedding for 'dist_token' and 'cls_token'
        checkpoint["pos_embed"][0, [0, 1], :] = checkpoint["pos_embed"][0, [1, 0], :]
        model.load_state_dict(checkpoint)
    return model
