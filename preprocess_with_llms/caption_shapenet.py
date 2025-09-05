import os
import io
import base64
import time
import json
from typing import Optional

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from superdec.data.dataloader import ShapeNet, SHAPENET_CATEGORIES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def render_pointcloud_image(points: np.ndarray, elev: float = 20.0, azim: float = 40.0, size: int = 512, bgcolor: str = "white") -> Image.Image:
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)
    pts = points if isinstance(points, np.ndarray) else points.cpu().numpy()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(x, y, z, s=1, c='black', depthshade=False)
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range
    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())
    ax.set_xlim(mid_x - Xb, mid_x + Xb)
    ax.set_ylim(mid_y - Xb, mid_y + Xb)
    ax.set_zlim(mid_z - Xb, mid_z + Xb)
    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def call_openai_gpt_vision(prompt: str, image_b64: str, model: str, api_key: Optional[str] = None, timeout: float = 30.0, max_retries: int = 3) -> str:
    try:
        import openai
    except Exception as e:
        raise RuntimeError("The 'openai' package is required. Please install it with 'pip install openai'.") from e

    if api_key:
        openai.api_key = api_key

    mapped_model = model
    if model.strip().lower() in ["gpt-mini", "gpt4o-mini", "gpt-4o-mini"]:
        mapped_model = "gpt-4o-mini"

    last_err = None
    for _ in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model=mapped_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2,
                timeout=timeout,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"OpenAI API call failed after retries: {last_err}")


_HF_MODEL = None
_HF_TOKENIZER = None
_HF_DEVICE = None


def _load_hf_mistral(model_id: str, device_pref: str = "auto", dtype_pref: str = "auto", quantize: Optional[str] = None):
    global _HF_MODEL, _HF_TOKENIZER, _HF_DEVICE
    if _HF_MODEL is not None and _HF_TOKENIZER is not None:
        return _HF_MODEL, _HF_TOKENIZER, _HF_DEVICE

    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise RuntimeError("The 'transformers' package is required. Install with 'pip install transformers'.") from e

    use_cuda = torch.cuda.is_available()
    if device_pref == "cuda" and not use_cuda:
        print("CUDA requested but not available; falling back to CPU.")
    device = "cuda" if (device_pref in ["auto", "cuda"]) and use_cuda else "cpu"

    torch_dtype = None
    if dtype_pref == "auto":
        if device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, dtype_pref)

    model_kwargs = dict(trust_remote_code=True)
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch_dtype

    if quantize is not None:
        # Optional quantization via bitsandbytes if available
        try:
            from transformers import BitsAndBytesConfig
            if quantize == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif quantize == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                print(f"Unknown quantize option '{quantize}', ignoring.")
        except Exception:
            print("bitsandbytes not available; loading without quantization.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    model.eval()
    if device == "cuda" and quantize is None:
        model.to(device)

    _HF_MODEL, _HF_TOKENIZER, _HF_DEVICE = model, tokenizer, device
    return model, tokenizer, device


@torch.no_grad()
def get_hf_mistral_embedding(text: str, model_id: str, device_pref: str = "auto", dtype_pref: str = "auto", max_length: int = 256, pooling: str = "mean", quantize: Optional[str] = None) -> np.ndarray:
    model, tokenizer, device = _load_hf_mistral(model_id, device_pref, dtype_pref, quantize)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Get hidden states from the base model (no LM head needed)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Use last_hidden_state for embedding pooling
    hidden = outputs.last_hidden_state  # [B, T, H]
    mask = attention_mask.unsqueeze(-1)  # [B, T, 1]

    if pooling == "mean":
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        emb = summed / counts
    elif pooling == "cls":
        # Use first token representation (BOS)
        emb = hidden[:, 0]
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    emb = emb.squeeze(0).float().cpu().numpy()
    return emb


def build_prompt(class_name: str) -> str:
    return (
        f"Provide a caption describing the {class_name}. "
        "Provide a single sentence and keep the caption simple. "
        "i.e. number of parts, part-names and geometrically significant attributes."
    )


@hydra.main(config_path="../configs", config_name="caption_shapenet", version_base=None)
def main(cfg: DictConfig):
    split = cfg.run.split
    tmp_cfg = DictConfig({
        "shapenet": {
            "path": cfg.shapenet.path,
            "categories": cfg.shapenet.categories,
            "normalize": cfg.shapenet.normalize,
        },
        "trainer": {
            "augmentations": False,
        },
    })

    dataset = ShapeNet(split=split, cfg=tmp_cfg)
    out_root = cfg.io.out_root
    _ensure_dir(out_root)

    openai_key = os.environ.get(cfg.captioner.openai_api_key_env, None)
    mistral_key = os.environ.get(cfg.embedding.api_key_env, None)

    for idx, sample in enumerate(dataset):
        model_info = dataset.models[idx]
        cat_id = model_info['category']
        cat_name = SHAPENET_CATEGORIES.get(cat_id, cat_id)
        model_id = model_info['model_id']

        out_dir = os.path.join(out_root, cat_id, model_id)
        _ensure_dir(out_dir)
        caption_path = os.path.join(out_dir, "caption.txt")
        embedding_path = os.path.join(out_dir, "embedding.npy")
        render_path = os.path.join(out_dir, "render.png")
        meta_path = os.path.join(out_dir, "meta.json")

        if (not cfg.io.overwrite) and os.path.exists(caption_path) and os.path.exists(embedding_path):
            continue

        pts = sample["points"].cpu().numpy().astype(np.float32)
        img = render_pointcloud_image(
            pts,
            elev=cfg.render.elev,
            azim=cfg.render.azim,
            size=cfg.render.size,
            bgcolor=cfg.render.bgcolor,
        )
        img.save(render_path)
        img_b64 = image_to_b64(img)

        prompt = build_prompt(cat_name)
        caption = call_openai_gpt_vision(
            prompt=prompt,
            image_b64=img_b64,
            model=cfg.captioner.gpt_model,
            api_key=openai_key,
            timeout=cfg.captioner.timeout,
            max_retries=cfg.captioner.max_retries,
        )

        emb = get_hf_mistral_embedding(
            caption,
            model_id=cfg.embedding.model_id,
            device_pref=cfg.embedding.device,
            dtype_pref=cfg.embedding.dtype,
            max_length=cfg.embedding.max_length,
            pooling=cfg.embedding.pooling,
            quantize=(cfg.embedding.quantize if 'quantize' in cfg.embedding else None),
        )

        with open(caption_path, 'w') as f:
            f.write(caption + "\n")
        np.save(embedding_path, emb)

        meta = {
            "category_id": cat_id,
            "category_name": cat_name,
            "model_id": model_id,
            "caption_model": cfg.captioner.gpt_model,
            "embedding_model": cfg.embedding.model_id,
            "points_normalized": bool(cfg.shapenet.normalize),
            "render": {
                "elev": cfg.render.elev,
                "azim": cfg.render.azim,
                "size": cfg.render.size,
                "bgcolor": cfg.render.bgcolor,
            },
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f)


if __name__ == "__main__":
    main()
