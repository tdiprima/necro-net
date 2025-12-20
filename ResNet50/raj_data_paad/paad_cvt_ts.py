# Convert to torchscript
import torch
import torch.nn as nn
import torchvision.models as models

# Load your trained model
model = models.resnet50(weights=None)

# Load checkpoint
checkpoint = torch.load("./local_data/output/paad_resnet50_best.pth")

# Match the fc layer structure from training
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, 10)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Convert to TorchScript
test_input = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "./local_data/output/paad_resnet50_best.pt")

print("✅ Model successfully converted to TorchScript")
