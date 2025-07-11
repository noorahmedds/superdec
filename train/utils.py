import torch
from torch.utils.data import DataLoader
from superdec.superdec import SuperDec
from superdec.data.dataloader import ShapeNet
from superdec.loss.loss import Loss
from torch.optim import Adam
import random
import numpy as np
import hydra

def build_model(cfg):
    model = SuperDec(cfg.superdec)
    return model.cuda() if torch.cuda.is_available() else model

def build_optimizer(cfg, model):
    return Adam(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)

def build_scheduler(cfg, optimizer, step_per_epoch):
    if not cfg.optimizer.enable_scheduler:
        return None
    cfg.scheduler.steps_per_epoch = step_per_epoch
    return hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

def build_dataloaders(cfg):
    if cfg.dataset == 'shapenet':
        train_ds = ShapeNet(split='train', cfg=cfg)
        val_ds = ShapeNet(split='val', cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset}")

    train_loader = DataLoader(train_ds, batch_size=cfg.trainer.batch_size, shuffle=True,
                              num_workers=cfg.trainer.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.trainer.batch_size, shuffle=False,
                            num_workers=cfg.trainer.num_workers, pin_memory=True)
    return {'train': train_loader, 'val': val_loader}


def build_loss(cfg):
    return Loss(cfg.loss)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
