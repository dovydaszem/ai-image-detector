import torch
import numpy as np
from PIL import Image

# Load traced model
model = torch.jit.load("model.pt")
model.eval()

def preprocess(img, dims=256):
    # Convert to greyscale
    img = img.convert("L")
    
    w, h = img.size
    
    if w < h:
        new_w = dims
        new_h = int(h * (dims / w))
    else:
        new_h = dims
        new_w = int(w * (dims / h))
    
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Center crop
    left = (new_w - dims) // 2
    top = (new_h - dims) // 2
    right = left + dims
    bottom = top + dims
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, normalise, and convert to tensor
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).float()
    
    # [H, W] -> [1, 1, H, W]
    return img_tensor.unsqueeze(0).unsqueeze(0)
