import torch
import torch.nn as nn
import torch.nn.functional as F


def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape
    inputs = inputs.reshape(b, c, h * w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1, 2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5 * (1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum() + 1)) + 0.5 * torch.sum(
        neg_mask * inputs_cos) / (neg_mask.sum() + 1)
    return loss


def get_seg_loss(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred, bg_label.type(torch.long)).sum() / (bg_sum + 1e-6)  # fix nan issue

    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred, fg_label.type(torch.long)).sum() / (fg_sum + 1e-6)  # fix nan issue

    return (bg_loss + fg_loss) * 0.5
