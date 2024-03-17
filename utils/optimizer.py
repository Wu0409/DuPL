import torch

import numpy as np

class CosWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8,)

        self.global_step = 0
        self.warmup_iter = np.float(warmup_iter)
        self.warmup_ratio = warmup_ratio
        self.max_iter = np.float(max_iter)
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = self.global_step / self.warmup_iter
            lr_add = (1 - self.global_step / self.warmup_iter) * self.warmup_ratio
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult + lr_add

        elif self.global_step < self.max_iter: 

            lr_mult = np.cos((self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter) * np.pi) * 0.5 + 0.5
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8,)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter: 

            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyWarmupSGD(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, momentum=0.9, weight_decay=weight_decay,)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_lr = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = (1 - self.global_step / self.warmup_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult * 10

        elif self.global_step < self.max_iter: 

            lr_mult = (1 - (self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1