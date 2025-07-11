import os 
import numpy as np
from tqdm import tqdm
import torch

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    def __init__(self, model, optimizer, scheduler, dataloaders, loss_fn, ctx, wandb_run=None, start_epoch=0, best_val_loss=float('inf')):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.ctx = ctx
        self.num_epochs = ctx.num_epochs
        self.save_path = ctx.save_path
        self.wandb_run = wandb_run
        self.best_val_loss = best_val_loss
        self.start_epoch = start_epoch


    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint and log to wandb."""
        if self.save_path is None:
            return
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'epoch': epoch,
            'val_loss': val_loss
        }
        
        ckpt_filename = f'epoch_{epoch+1}.pt'
        ckpt_path = os.path.join(self.save_path, ckpt_filename)
        torch.save(checkpoint, ckpt_path)
        
        # Log checkpoint as wandb artifact
        if self.wandb_run is not None and wandb is not None:
            artifact = wandb.Artifact(ckpt_filename, type='model')
            artifact.add_file(ckpt_path)
            self.wandb_run.log_artifact(artifact)

    @torch.no_grad()
    def evaluate(self, epoch):
        """Evaluate model on validation set."""
        self.model.eval()
        loader = self.dataloaders['val']
        pbar = tqdm(loader, desc=f"Eval  {epoch+1}/{self.num_epochs}", leave=False)

        total_loss = 0.0
        total_batches = 0
        avg_loss_dict = {}
        all_outputs = {
            'names': [], 'pc': [], 'assign_matrix': [], 'scale': [], 'rotation': [],
            'translation': [], 'exponents': [], 'exist': []
        }

        for batch in pbar:
            pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
            outdicts = self.model(pc)
            loss, loss_dict = self.loss_fn(pc, normals, outdicts[-1])

            total_loss += loss.item()
            total_batches += 1
            
            # Accumulate loss components
            for k, v in loss_dict.items():
                avg_loss_dict[k] = avg_loss_dict.get(k, 0.0) + v

            pbar.set_postfix({k: f"{v / total_batches:.4f}" for k, v in avg_loss_dict.items()})

        # Compute averages
        for k in avg_loss_dict:
            avg_loss_dict[k] /= total_batches

        return avg_loss_dict

    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        loader = self.dataloaders['train']
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

        total_loss = 0.0
        total_batches = 0
        avg_loss_dict = {}

        for batch in pbar:
            pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
            outdicts = self.model(pc)
            loss, loss_dict = self.loss_fn(pc, normals, outdicts[-1])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_batches += 1
            for k, v in loss_dict.items():
                avg_loss_dict[k] = avg_loss_dict.get(k, 0.0) + v

            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

        # Compute averages
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        for k in avg_loss_dict:
            avg_loss_dict[k] /= total_batches

        # Log training metrics to wandb
        if self.wandb_run is not None:
            log_dict = {"train/loss": avg_loss}
            log_dict.update({f"train/{k}": v for k, v in avg_loss_dict.items()})
            
            # Log learning rate
            if self.optimizer.param_groups:
                lr = self.optimizer.param_groups[0].get('lr', None)
                if lr is not None:
                    log_dict["train/lr"] = lr
                    
            self.wandb_run.log(log_dict, step=epoch)

    def train(self):
        """Main training loop."""
        save_every = getattr(self.ctx, 'save_every_n_epochs', 1)
        
        for epoch in range(self.start_epoch, self.num_epochs):
            # Training phase
            self.train_one_epoch(epoch)
            
            # Evaluation phase (every epoch)
            do_save = ((epoch + 1) % save_every == 0) or (epoch == self.num_epochs - 1)
            val_metrics = self.evaluate(epoch)
            val_loss = val_metrics.get('loss', None) or list(val_metrics.values())[0]
            
            # Log validation metrics to wandb (every epoch)
            if self.wandb_run is not None:
                self.wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
            
            # Save best checkpoint whenever validation loss improves
            if do_save: # val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)