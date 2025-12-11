import torch
import torch.nn as nn
import torchvision.models as models

# Load your trained model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)  # match your 10 classes

model.load_state_dict(torch.load("../models/DecaResNet_v2.pth"))
model.eval()

# Ensure model accepts 224x224x3 input (standard for WSI patches)
test_input = torch.randn(1, 3, 224, 224)
output = model(test_input)
print(f"Model output shape: {output.shape}")  # Should be [1, num_classes]

try:
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, "DecaResNet_v2.pt")
    print("✅ Successfully scripted model")
except Exception as e:
    print(f"❌ Script failed: {e}")
    # Method 2: Trace (fallback)
    traced_model = torch.jit.trace(model, test_input)
    torch.jit.save(traced_model, "DecaResNet_v2.pt")
    print("✅ Successfully traced model")

# Validate the converted model
loaded_model = torch.jit.load("../models/DecaResNet_v2.pt")
with torch.no_grad():
    original_output = model(test_input)
    converted_output = loaded_model(test_input)
    assert torch.allclose(original_output, converted_output, atol=1e-5)
    print("✅ Model conversion validated")
