import io
import torch
import base64
import numpy as np
from PIL import Image
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from matplotlib.colors import LinearSegmentedColormap

# custom colormap with transparency
trans_jet = LinearSegmentedColormap.from_list(
    "trans_jet",
    [
        (0.0, (1.0, 1.0, 1.0, 0.0)),
        (0.3, (1.0, 1.0, 1.0, 0.0)),
        (0.5, (0.0, 0.0, 1.0, 0.5)),
        (0.6, (0.0, 1.0, 0.9, 1.0)),
        (0.7, (0.0, 1.0, 0.0, 1.0)),
        (0.8, (1.0, 1.0, 0.0, 1.0)),
        (0.9, (1.0, 0.5, 0.0, 1.0)),
        (1.0, (0.6, 0.0, 0.0, 1.0)),
    ],
)

class CNN(nn.Module):
    def __init__(self):
        # super() in Python calls methods of a parent class
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv4 = nn.Conv2d(64, 128, 3, padding="same")

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.mpool = nn.MaxPool2d(2, 2)
        self.apool = nn.AdaptiveAvgPool2d((4,4))
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.mpool(F.relu(self.bn1(self.conv1(x))))
        x = self.mpool(F.relu(self.bn2(self.conv2(x))))
        x = self.mpool(F.relu(self.bn3(self.conv3(x))))
        x = self.mpool(F.relu(self.bn4(self.conv4(x))))

        x = self.apool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)

# Recreate model architecture
model = CNN()
model.load_state_dict(torch.load(Path(__file__).parent / "model_weights.pth", map_location="cpu"))
model.eval()

def preprocess(img, dims=256):
    # Convert to greyscale
    img = img.convert("L")
    w, h = img.size
    
    scale = dims / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Pad the image to be square
    padded = Image.new("L", (dims, dims))
    left = (dims - new_w) // 2
    top = (dims - new_h) // 2
    padded.paste(img, (left, top))

    # Convert to numpy and normalise
    img_np = np.array(padded, dtype=np.float32) / 255.0

    # Convert to tensor and change dims [H, W] -> [1, 1, H, W]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    return img_tensor

def apply_gradcam(model, img_tensor, w, h, dims=256):
    # Apply Grad-CAM to 256x256 input
    gradcam = GradCAMPlusPlus(model=model, target_layers=[model.conv4])
    heatmap = gradcam(input_tensor=img_tensor, targets=[lambda x: x])[0]

    # Remove padding, and resize the heatmap to the original image size
    scale = dims / max(w, h)
    nw = int(w * scale)
    nh = int(h * scale)
    left = int((dims - nw) // 2)
    top = int((dims - nh) // 2)

    heatmap = heatmap[top:top+nh, left:left+nw]
    heatmap = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.Resampling.BILINEAR)
    
    # applying colormap and forcing conversion to numpy array
    colored_heatmap = np.array(trans_jet(heatmap))
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    # create RGBA image and resize to original dimensions
    heatmap_img = Image.fromarray(colored_heatmap, mode="RGBA").resize((w, h), Image.Resampling.BILINEAR)
    
    # encode to base64 string
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
