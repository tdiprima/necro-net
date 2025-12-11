import torch
import torch.onnx
import torchvision.models as models

# Create a ResNet model with the correct number of classes
# Option A: If you used a standard torchvision ResNet (e.g., ResNet50)
model = models.resnet50(pretrained=False)  # or resnet18, resnet34, resnet101, etc.

# Modify the final layer to have 10 classes (matching your checkpoint)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)  # 10 classes instead of 1000

# Load the saved weights
checkpoint = torch.load("../models/DecaResNet_v2.pth")
model.load_state_dict(checkpoint)  # Now the dimensions should match
model.eval()  # Set to evaluation mode

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust dimensions if needed

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "../models/DecaResNet_v2.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print("Model successfully converted to ONNX format!")
