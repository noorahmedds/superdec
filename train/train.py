import os

import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from utils import build_model, build_optimizer, build_scheduler, build_dataloaders, build_loss, set_seed
import torch
try:
    import wandb
except ImportError:
    wandb = None

def to_str_dict(d):
    if isinstance(d, dict):
        return {str(k): to_str_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_str_dict(i) for i in d]
    else:
        return d

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Set seeds for reproducibility
    set_seed(cfg.seed)
    print(f"Set seed to {cfg.seed}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run = None
    if cfg.use_wandb and wandb is not None:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.run_name
        )

    model = build_model(cfg).to(device)

    dataloaders = build_dataloaders(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(dataloaders['train'])) # None if disabled

    loss_fn = build_loss(cfg).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    checkpoint_path = getattr(cfg.checkpoints.resume_from, 'resume_from', None)
    if cfg.checkpoints.resume_from is not None:
        checkpoint = torch.load(cfg.checkpoints.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if cfg.checkpoints.keep_epoch:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}, best_val_loss={best_val_loss}")

    cfg.trainer.save_path = os.path.join(cfg.trainer.save_path, cfg.run_name)
    if not os.path.exists(cfg.trainer.save_path):
        os.makedirs(cfg.trainer.save_path, exist_ok=True)
    # save cfg in save path as yaml
    with open(os.path.join(cfg.trainer.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(cfg, f=fp)

    if run is not None:
        artifact = wandb.Artifact(name="config", type="config")
        artifact.add_file(os.path.join(cfg.trainer.save_path, "config.yaml"))
        run.log_artifact(artifact)

    trainer = Trainer(model, optimizer, scheduler, dataloaders, loss_fn, cfg.trainer, run, start_epoch=start_epoch, best_val_loss=best_val_loss)
    trainer.train()
    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()