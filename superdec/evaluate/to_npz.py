import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.utils.evaluation import get_outdict
from superdec.loss.loss import Loss
from superdec.data.dataloader import ShapeNet
from typing import Dict, Any
from tqdm import tqdm

def main(cfg: DictConfig) -> None:
    """
    Main evaluation entrypoint. Loads config, runs evaluation, prints results.
    """

    device = cfg.get('device', 'cuda')
    # Dataloader
    dataset = ShapeNet(split=cfg.dataloader.split, cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers)
    ckp_path = os.path.join(cfg.checkpoints_folder, f'epoch_{str(cfg.epoch)}.pt')
    config_path = os.path.join(cfg.checkpoints_folder, cfg.config_file)
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)

    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        for i, b in tqdm(enumerate(dataloader)):
            points = b['points'].to(device).float()
            outdict_list = model(points)
            outdict = outdict_list[-1]  # Use last layer output
            names = b.get('model_id', np.arange(points.shape[0]))
            if i == 0:
                pred_handler = PredictionHandler.from_outdict(outdict, points, names)
            else:
                pred_handler.append_outdict(outdict, points, names)

    pred_handler.save_npz(os.path.join(cfg.checkpoints_folder, f'{str(cfg.epoch)}_{cfg.dataloader.split}.npz')) # this step takes a lot of time (~1 minute)


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../configs", config_name="save_npz")
    def run_main(cfg: DictConfig):
        main(cfg)
    run_main()