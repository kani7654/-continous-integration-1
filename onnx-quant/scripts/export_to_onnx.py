import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MODEL_PATH = "model.onnx"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_dummy_model(path=MODEL_PATH):
    model = SimpleCNN()
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Exported ONNX model to {path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    create_dummy_model()
