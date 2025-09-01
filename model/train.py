# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from drepavit.core import *  # your model/loops live here

def main():
    p = argparse.ArgumentParser(description="Train (anonymous skeleton)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="weights")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    if 'train' in globals():
        train(epochs=args.epochs, lr=args.lr, out_dir=args.out)   # noqa: F405
    else:
        print("[WARN] No 'train(...)' found in drepavit.core. Add one or edit model/train.py.")

if __name__ == "__main__":
    main()
