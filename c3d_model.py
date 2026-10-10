import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    """
    C3D (Convolutional 3D Network) spatiotemporal feature extractor
    Matches Tran et al. architecture and Keras weights in weights.h5.
    Input shape: (Batch, 3, 16, 112, 112)
    """
    def __init__(self, num_classes=487):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool1(h)

        h = F.relu(self.conv2(h))
        h = self.pool2(h)

        h = F.relu(self.conv3a(h))
        h = F.relu(self.conv3b(h))
        h = self.pool3(h)

        h = F.relu(self.conv4a(h))
        h = F.relu(self.conv4b(h))
        h = self.pool4(h)

        h = F.relu(self.conv5a(h))
        h = F.relu(self.conv5b(h))
        h = F.pad(h, (0, 1, 0, 1, 0, 0))
        h = self.pool5(h)

        h = h.permute(0, 2, 3, 4, 1).contiguous().view(h.size(0), -1)

        f6 = F.relu(self.fc6(h))
        f7 = F.relu(self.fc7(f6))
        logits = self.fc8(f7)
        return logits, f7

    def extract_features(self, x):
        """Returns 4096-dimensional fc7 feature representation"""
        _, f7 = self.forward(x)
        return f7


def load_c3d_model(weights_path="weights.h5", device="cpu"):
    """
    Loads C3D model and transfers weights from Keras weights.h5
    """
    model = C3D(num_classes=487)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    with h5py.File(weights_path, 'r') as f:
        # Load Conv3D layers
        for layer_name in ['conv1', 'conv2', 'conv3a', 'conv3b', 'conv4a', 'conv4b', 'conv5a', 'conv5b']:
            w = f[layer_name][f'{layer_name}/kernel:0'][:]
            b = f[layer_name][f'{layer_name}/bias:0'][:]
            w_torch = torch.from_numpy(w).permute(4, 3, 0, 1, 2)
            b_torch = torch.from_numpy(b)
            layer = getattr(model, layer_name)
            layer.weight.data.copy_(w_torch)
            layer.bias.data.copy_(b_torch)

        # Load Linear layers
        for layer_name in ['fc6', 'fc7', 'fc8']:
            w = f[layer_name][f'{layer_name}/kernel:0'][:]
            b = f[layer_name][f'{layer_name}/bias:0'][:]
            w_torch = torch.from_numpy(w).t()
            b_torch = torch.from_numpy(b)
            layer = getattr(model, layer_name)
            layer.weight.data.copy_(w_torch)
            layer.bias.data.copy_(b_torch)

    model.to(device)
    model.eval()
    return model


def get_model_summary(weights_path="weights.h5"):
    """
    Inspects weights.h5 and returns layer metrics and parameter counts
    """
    summary = []
    total_params = 0
    if not os.path.exists(weights_path):
        return summary, total_params

    with h5py.File(weights_path, 'r') as f:
        layer_names = f.attrs.get('layer_names', [])
        for l in layer_names:
            name = l.decode('utf-8') if isinstance(l, bytes) else l
            grp = f[name]
            w_names = grp.attrs.get('weight_names', [])
            layer_params = 0
            shapes = []
            for wn in w_names:
                wn_str = wn.decode('utf-8') if isinstance(wn, bytes) else wn
                dset = grp[wn_str]
                p_count = int(np.prod(dset.shape))
                layer_params += p_count
                shapes.append(f"{wn_str.split('/')[-1]}: {list(dset.shape)}")
            total_params += layer_params
            summary.append({
                "Layer Name": name,
                "Weights & Shapes": ", ".join(shapes) if shapes else "Non-parametric",
                "Parameters": layer_params
            })
    return summary, total_params
