# -*- coding: utf-8 -*-
import argparse, torch
from PIL import Image
import numpy as np

def load_image(path, size=224):
    im = Image.open(path).convert("RGB").resize((size,size))
    arr = np.asarray(im).astype("float32")/255.0
    arr = arr.transpose(2,0,1)[None,...]  # 1x3xHxW
    return torch.from_numpy(arr)

def main():
    p = argparse.ArgumentParser(description="Inference on one image (TorchScript)")
    p.add_argument("--weights", required=True, help="TorchScript file (.ts)")
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    m = torch.jit.load(args.weights, map_location="cpu")
    x = load_image(args.image, size=args.size)
    with torch.inference_mode():
        y = m(x)
    print("logits/probs:", y.squeeze().tolist())

if __name__ == "__main__":
    main()
