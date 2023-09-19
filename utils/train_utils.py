from pytorch_lightning import LightningModule
import torch
from torchmetrics import Accuracy
import numpy as np
import torch.nn as nn

from optimizers.optimizers import get_optimizer
from schedulers.schedulers import get_scheduler
from criterions.criterions import get_criterion


def get_plmodule(model, args):
    model = Model(model, args)
    return model


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Model(LightningModule):
    def __init__(self, model, args):
        super().__init__()

        self.model = model
        self.epochs = args.epochs
        self.criterion = get_criterion(args)
        self.optimizer = get_optimizer(model, args)
        self.scheduler = get_scheduler(self.optimizer, args)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=args.num_classes)
        self.cutmix_beta = 1
        self.cutmix_prob = args.cutmix_prob

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(x.size(0)).to(self.device)
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            out = self.model(x)
            loss = self.criterion(out, target_a) * lam + self.criterion(
                out, target_b
            ) * (1.0 - lam)

        else:
            out = self.model(x)
            loss = self.criterion(out, y)
        preds = torch.argmax(out, dim=1)
        self.train_accuracy.update(preds, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train/acc", self.train_accuracy, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val/acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        # self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "test/acc", self.test_accuracy, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
