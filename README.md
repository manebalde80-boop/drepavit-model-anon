# DrepaViT (anonymous) – Model Code

# DrepaViT — Code pour l'article

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Release](https://img.shields.io/github/v/tag/manebalde80-boop/drepavit-model-anon)


This repository contains the *blind* version of the DrepaViT model code
(training + TorchScript export) used in our paper's demo.

## Quickstart

pip install -r requirements.txt
python model/train.py --help
python model/export_torchscript.py --ckpt path/to/checkpoint.pt --out weights/drepavit.ts
python inference/predict.py --weights weights/drepavit.ts --image samples/patch_001.png

python -m venv .venv && source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

# (weights non fournis) Voir scripts/prepare_weights.md
# Exemple d’inférence TorchScript (une image ou dossier):
python inference/predict.py \
  --weights /path/to/model.torchscript.pt \
  --input /path/to/image_or_dir \
  --out results.json

## Weights
See `scripts/prepare_weights.md` (anonymous link & SHA256).

## License
MIT – "The DrepaViT Authors".
