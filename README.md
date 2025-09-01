# DrepaViT (anonymous) – Model Code

This repository contains the *blind* version of the DrepaViT model code
(training + TorchScript export) used in our paper's demo.

## Quickstart
```bash
pip install -r requirements.txt
python model/train.py --help
python model/export_torchscript.py --ckpt path/to/checkpoint.pt --out weights/drepavit.ts
python inference/predict.py --weights weights/drepavit.ts --image samples/patch_001.png
```

## Weights
See `scripts/prepare_weights.md` (anonymous link & SHA256).

## License
MIT – "The DrepaViT Authors".
