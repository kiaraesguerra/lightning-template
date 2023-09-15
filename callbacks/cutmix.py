from functools import partial
from pytorch_lightning.callbacks import Callback
import torch
import torch.nn as nn
import numpy as np

from criterions.criterions import get_criterion


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix(Callback):
    def __init__(
        self,
        cutmix_beta: float = 1.0,
        cutmix_prob: float = 0.5,
        criterion: nn.Module = None,
    ):
        self.cutmix_beta = cutmix_beta
        self.cutmix_prob = cutmix_prob
        self.criterion = criterion

    def training_step(self, batch, batch_idx):
        img, label = batch
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2])
            )
            # compute output
            with torch.cuda.amp.autocast():
                out = self.model(img)
                breakpoint(0)
                loss = self.criterion(out, target_a) * lam + self.criterion(
                    out, target_b
                ) * (1.0 - lam)

            return loss


def cutmix_callback(args):
    cutmix = CutMix(cutmix_beta=1.0, cutmix_prob=0.5, criterion=get_criterion(args))

    return cutmix
