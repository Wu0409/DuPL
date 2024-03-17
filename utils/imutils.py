import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from torchvision import datasets, transforms
from utils import randomaug


def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16), :]


def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:, 0, :, :] = imgs[:, 0, :, :] * std[0] + mean[0]
    _imgs[:, 1, :, :] = imgs[:, 1, :, :] * std[1] + mean[1]
    _imgs[:, 2, :, :] = imgs[:, 2, :, :] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs


def denormalize_img2(imgs=None):
    # _imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0


def minmax_norm(x):
    for i in range(x.shape[0]):
        x[i, ...] = x[i, ...] - x[i, ...].min()
        x[i, ...] = x[i, ...] / x[i, ...].max()
    return x


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


# 定义类别序号对应的颜色映射
voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


# 将类别序号矩阵转换为彩色分割结果
def convert_class_to_color(class_matrix):
    h, w = class_matrix.shape
    # Create an empty RGB image with the given size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Map each class index to a color from the PASCAL VOC color map
    for i in range(1, class_matrix.max() + 1):
        img[class_matrix == i] = voc_colors[i % len(voc_colors)]
    return img


def get_rand_aug():
    aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return aug


def augment_data(images, labels):
    for i in range(images.shape[0]):
        augment = get_rand_aug()
        images[i] = augment(images[i])
        if random.random() < 0.5:  # Random Flipping
            images[i] = torch.flip(images[i], dims=[2])
            labels[i] = torch.flip(labels[i], dims=[2])

    return images, labels


# Random mask
def random_mask(input_data, mask_prob):
    n, _, h, w = input_data.shape
    h /= 32
    w /= 32

    mask = torch.rand((n, int(h), int(w)), device=input_data.device) > mask_prob
    mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
    mask = F.interpolate(mask.float(), scale_factor=32, mode='nearest')
    masked_input = input_data * mask.float()
    return masked_input


def high_cam_mask(input_data, cams, weight=8.0, sigma=0.25):
    n, _, h, w = cams.shape
    cam_h, _ = torch.max(cams, dim=1, keepdim=True)
    cam_h = cam_h.squeeze(1)  # 每个位置的最高激活值

    # cam_h[cam_h < 0.45] = 0

    mask = torch.sigmoid((cam_h - sigma) * weight).unsqueeze(1)
    mask = F.interpolate(mask, scale_factor=16, mode="bilinear", align_corners=False)

    # plt.imshow(mask[0][0].cpu().numpy())
    # plt.show()

    inputs_m = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_data - input_data * mask)

    return inputs_m


# For v4
def high_cam_mask_1(input_data, cams, cls_label, mask_prob=1.0, k=1.0, low_thres=0.5):
    n, _, h, w = cams.shape
    cams_s = F.interpolate(cams, size=(h // 16, w // 16), mode='bilinear', align_corners=True)
    cam_h_mask, _ = cams_s.data.max(1)  # 每个位置的最高激活值

    # plt.imshow(cam_h_mask[0].cpu().numpy())
    # plt.show()

    mask = []
    for i in range(n):
        cams_s_i = cams_s[i]
        topk_mask_i_c = torch.ones((h // 16) * (w // 16), device=input_data.device)
        for c in (torch.nonzero(cls_label[0])):
            cams_s_i_c = cams_s_i[c].squeeze(0)
            valid_num = torch.sum((cams_s_i_c > low_thres).int())

            # select top k
            cam_flat = cams_s_i_c.view(-1)
            _, topk_indices = torch.topk(cam_flat, k=int(valid_num * k))

            # random mask
            mask_pos_num = int(topk_indices.shape[-1] * mask_prob)
            mask_indices = torch.randperm(mask_pos_num)[:mask_pos_num]
            selected_pos = topk_indices[mask_indices]

            topk_mask_i_c[selected_pos] = 0

        topk_mask_i_c = topk_mask_i_c.view(h // 16, w // 16).unsqueeze(0)
        mask.append(topk_mask_i_c)

    topk_mask = F.interpolate(torch.cat(mask, dim=0).float().unsqueeze(1), scale_factor=16, mode='nearest')

    inputs_m = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(input_data * topk_mask)

    return inputs_m


# For v5
def generate_token_mask(cams, cls_label, mask_prob=0.7, k=0.3, low_thres=0.5):
    n, _, h, w = cams.shape
    cams_s = F.interpolate(cams, size=(h // 16, w // 16), mode='bilinear', align_corners=True)
    cam_h_mask, _ = cams_s.data.max(1)  # 每个位置的最高激活值

    # plt.imshow(cam_h_mask[0].cpu().numpy())
    # plt.show()

    mask = []
    for i in range(n):
        cams_s_i = cams_s[i]
        topk_mask_i_c = torch.ones((h // 16) * (w // 16), device=cams.device)
        for c in (torch.nonzero(cls_label[0])):
            cams_s_i_c = cams_s_i[c].squeeze(0)
            valid_num = torch.sum((cams_s_i_c > low_thres).int())

            # select top k
            cam_flat = cams_s_i_c.view(-1)
            _, topk_indices = torch.topk(cam_flat, k=int(valid_num * k))

            # random mask
            mask_pos_num = int(topk_indices.shape[-1] * mask_prob)
            mask_indices = torch.randperm(mask_pos_num)[:mask_pos_num]
            selected_pos = topk_indices[mask_indices]

            topk_mask_i_c[selected_pos] = 0

        topk_mask_i_c = topk_mask_i_c.view(h // 16, w // 16).unsqueeze(0)
        mask.append(topk_mask_i_c)

    token_mask = torch.cat(mask, dim=0).float()

    return token_mask


# for v7
def high_cam_mask_2(input_data, cams, cls_label, mask_prob=1.0, k=1.0, low_thres=0.5):
    n, _, h, w = cams.shape
    cams_s = F.interpolate(cams, size=(h // 16, w // 16), mode='bilinear', align_corners=True)
    cam_bg = 1 - cams_s.data.max(1)[0]  # 每个位置的最高激活值

    mask = []
    for i in range(n):
        cams_s_i = cams_s[i]
        topk_mask_i_c = torch.ones((h // 16) * (w // 16), device=input_data.device)
        for c in (torch.nonzero(cls_label[0])):
            cams_s_i_c = cams_s_i[c].squeeze(0)
            valid_num = torch.sum((cams_s_i_c > low_thres).int())

            # skip masking small objects
            if valid_num < 20:
                continue

            # select top k (class)
            cam_flat = cams_s_i_c.view(-1)
            _, topk_indices = torch.topk(cam_flat, k=int(valid_num * k))

            # random mask
            mask_pos_num = topk_indices.shape[-1]
            mask_indices = torch.randperm(mask_pos_num)[:int(mask_pos_num * mask_prob)]
            selected_pos = topk_indices[mask_indices]

            topk_mask_i_c[selected_pos] = 0

        # select top k (bg)
        cam_bg_flat_i = cam_bg[i].view(-1)
        valid_num = torch.sum((cam_bg_flat_i > low_thres).int())
        _, topk_indices = torch.topk(cam_bg_flat_i, k=int(valid_num * k))

        # random mask (bg)
        mask_pos_num = topk_indices.shape[-1]
        mask_indices = torch.randperm(mask_pos_num)[:int(mask_pos_num * mask_prob)]
        selected_pos = topk_indices[mask_indices]
        topk_mask_i_c[selected_pos] = 0

        topk_mask_i_c = topk_mask_i_c.view(h // 16, w // 16).unsqueeze(0)
        mask.append(topk_mask_i_c)

    mask_token = torch.cat(mask, dim=0).float().unsqueeze(1)  # token-level mask
    mask_image = F.interpolate(mask_token, scale_factor=16, mode='nearest')  # image-level mask (seg_loss)

    return mask_token, mask_image


def tensorboard_image(imgs=None, cam=None):
    _imgs = denormalize_img(imgs=imgs)
    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)

    cam = F.interpolate(cam, size=_imgs.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.cpu()
    cam_max = cam.max(dim=1)[0]
    cam_heatmap = plt.get_cmap('jet')(cam_max.numpy())[:, :, :, 0:3] * 255
    cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
    cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
    grid_cam = torchvision.utils.make_grid(tensor=cam_img.type(torch.uint8), nrow=2)

    return grid_imgs, grid_cam


def get_rand_aug():
    aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        # transforms.RandomGrayscale(p=0.2),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return aug


def flip_img_box(img_box, img_width):
    # 原始左上角和右下角坐标
    y_start, y_end, x_start, x_end = img_box

    # 计算水平翻转后的新坐标
    y_start_new = y_start
    y_end_new = y_end
    x_end_new = img_width - x_end
    x_start_new = img_width - x_start

    # 返回新的边界框坐标
    return torch.tensor([y_start_new, y_end_new, x_end_new, x_start_new])


def augment_data(images, img_box):
    img_box_new_list = []
    for i in range(images.shape[0]):
        augment = get_rand_aug()
        images[i] = augment(images[i])
        images[i] = torch.flip(images[i], dims=[2])
        img_box_new_list.append(flip_img_box(img_box[i], images.shape[3]))

    img_box = torch.stack(img_box_new_list, dim=0)
    return images, img_box


def augment_data_strong(images, n=4, m=20):
    trans_pil = transforms.ToPILImage()
    trans_tensor = transforms.ToTensor()
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for i in range(images.shape[0]):
        image_pil_i = trans_pil(images[i])
        aug = randomaug.RandAugment(n, m)
        image_pil_i_aug = trans_tensor(aug(image_pil_i))
        images[i] = norm(image_pil_i_aug)
        images[i] = torch.flip(images[i], dims=[2])

    return images
