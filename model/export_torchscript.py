# -*- coding: utf-8 -*-
import argparse, torch
from pathlib import Path
from drepavit.core import *  # noqa

def find_model_instance():
    import torch.nn as nn
    cand = [v for k,v in globals().items() if isinstance(v, type) and issubclass(v, nn.Module)]
    if cand:
        return cand[0]()
    if 'build_model' in globals():
        return build_model()  # noqa: F405
    raise RuntimeError("No model class or build_model() found in drepavit.core")

def main():
    ap = argparse.ArgumentParser(description="Export TorchScript")
    ap.add_argument("--ckpt", required=False, help="checkpoint .pt/.pth (optional, state_dict)")
    ap.add_argument("--out", required=True, help="output .ts")
    ap.add_argument("--shape", default="1,3,224,224", help="dummy input shape, e.g., 1,3,224,224")
    args = ap.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    model = find_model_instance()
    model.eval()

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)

    dummy = torch.randn(*shape)
    try:
        scripted = torch.jit.trace(model, dummy)
    except Exception:
        scripted = torch.jit.script(model)

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    scripted.save(args.out)
    print("[OK] TorchScript saved to", args.out)

if __name__ == "__main__":
    main()
