import os

import numpy as np
import torch
from torch.utils.data import Dataset

from superdec.data.transform import RotateAroundAxis3d, Scale3d, RandomMove3d, Compose


SHAPENET_CATEGORIES = {
    "04379243": "table", "02958343": "car", "03001627": "chair", "02691156": "airplane",
    "04256520": "sofa", "04090263": "rifle", "03636649": "lamp", "03691459": "loudspeaker",
    "02933112": "cabinet", "03211117": "display", "04401088": "telephone", "02828884": "bench",
    "04530566": "watercraft"
}

def normalize_points(points):
    translation = points.mean(0)
    points = points - translation
    scale = 2 * np.max(np.abs(points))
    points = points / scale
    return points, translation, scale


def get_transforms(split: str, cfg):
    if split != 'train' or not cfg.trainer.augmentations:
        return None

    return Compose([
        Scale3d(),
        RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 0, 1)),
        RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
        RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 1, 0)),
        RandomMove3d(
            x_min=-0.1, x_max=0.1,
            y_min=-0.05, y_max=0.05,
            z_min=-0.1, z_max=0.1
        ),
    ])


class ShapeNet(Dataset):
    def __init__(self, split: str, cfg):
        super().__init__()
        self.split = split
        self.data_root = cfg.shapenet.path

        self.transform = get_transforms(split, cfg)
        self.normalize = cfg.shapenet.normalize

        self.categories = self._load_categories(cfg.shapenet.categories)
        self.models = self._gather_models()

    def _load_categories(self, categories):
        if categories is None:
            return [d for d in os.listdir(self.data_root)
                    if os.path.isdir(os.path.join(self.data_root, d))]
        print(f"Categories for split '{self.split}': {', '.join(SHAPENET_CATEGORIES[c] for c in categories)}")
        return categories

    def _gather_models(self):
        models = []
        for c in self.categories:
            category_path = os.path.join(self.data_root, c)
            split_file = os.path.join(category_path, f'{self.split}.lst')
            if not os.path.exists(split_file):
                continue
            with open(split_file, 'r') as f:
                model_ids = [line.strip() for line in f if line.strip()]
            models.extend([{'category': c, 'model_id': m} for m in model_ids])
        return models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.data_root, model['category'], model['model_id'])
        

        if self.split == 'test':
            pc_data = np.load(os.path.join(model_path, "pointcloud_4096.npz"))
            points = pc_data["points"]
            normals = pc_data["normals"]
        else:
            pc_data = np.load(os.path.join(model_path, "pointcloud.npz"))
            n_points = pc_data["points"].shape[0]
            idxs = np.random.choice(n_points, 4096, replace=False)
            points = pc_data["points"][idxs]
            normals = pc_data["normals"][idxs]

        if self.normalize:
            points, translation, scale  = normalize_points(points)
        else:
            translation = np.zeros(3)
            scale = 1.0

        if self.transform is not None:
            t_data = self.transform(points=points, normals=normals)
            points = t_data['points']
            normals = t_data['normals']

        return {
            "points": torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "point_num": points.shape[0],
            "model_id": model['model_id']
        }

    def name(self):
        return 'ShapeNet'