import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from utils import build_model, build_optimizer, build_scheduler, build_dataloaders, build_loss
import torch
import random
import numpy as np
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
    
    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run = None
    if getattr(cfg, "use_wandb", False) and wandb is not None:
        # Convert DictConfig to a flat dict with string keys for wandb (only top-level keys)
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg_container, dict):
            config_dict = {str(k): v for k, v in cfg_container.items() if not isinstance(v, (dict, list))}
        else:
            config_dict = {}
        run_name = None
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "name"):
            run_name = cfg.wandb.name
        run = wandb.init(
            project="superdec",
            config=config_dict,
            name=run_name
        )

    model = build_model(cfg).to(device)

    dataloaders = build_dataloaders(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(dataloaders['train'])) # None if disabled

    loss_fn = build_loss(cfg).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if hasattr(cfg, 'checkpoints') and getattr(cfg.checkpoints, 'resume_from', None):
        checkpoint_path = cfg.checkpoints.resume_from
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}, best_val_loss={best_val_loss}")

    # Save hydra config to wandb as artifact if enabled
    if run is not None:
        from hydra import compose, initialize
        import os
        config_path = os.getcwd()
        config_file = os.path.join(config_path, '.hydra', 'config.yaml')
        if os.path.exists(config_file):
            run.save(config_file)

    trainer = Trainer(model, optimizer, scheduler, dataloaders, loss_fn, cfg.trainer, run, start_epoch=start_epoch, best_val_loss=best_val_loss)
    trainer.train()
    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()