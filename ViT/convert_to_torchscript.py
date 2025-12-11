"""
Converts a trained PyTorch ViT model to TorchScript format (.pt) for deployment.
TorchScript models can be used in production environments without Python dependencies.
uv run convert_to_torchscript.py --checkpoint ../models/vit_best.pth --num_classes 10 --output ../models/vit_model.pt

uv run convert_to_torchscript.py \
  --checkpoint ../models/vit_best.pth \
  --num_classes 10 \
  --output ../models/vit_model.pt \
  --verify
"""

import argparse

import torch
import torch.nn as nn
from rich_argparse import RichHelpFormatter
from torchvision import models


def load_vit_model(num_classes, device, checkpoint_path):
    """
    Loads the ViT model with custom head from checkpoint.
    """
    try:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    except Exception:
        model = models.vit_b_16(pretrained=True)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    model = model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def convert_to_torchscript(model, device, img_size=224, method="trace"):
    """
    Converts model to TorchScript using either trace or script method.

    Args:
        model: PyTorch model to convert
        device: device to run conversion on
        img_size: input image size (default 224 for ViT)
        method: 'trace' or 'script' - trace is recommended for ViT

    Returns:
        TorchScript model
    """
    model.eval()

    if method == "trace":
        # Create example input (batch_size=1, channels=3, height=img_size, width=img_size)
        example_input = torch.randn(1, 3, img_size, img_size).to(device)

        # Trace the model
        with torch.no_grad():
            scripted_model = torch.jit.trace(model, example_input)

        print(f"Model successfully traced with input shape: {example_input.shape}")

    elif method == "script":
        # Script the model (may not work for all architectures)
        scripted_model = torch.jit.script(model)
        print("Model successfully scripted")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")

    return scripted_model


def verify_torchscript_model(scripted_model, original_model, device, img_size=224):
    """
    Verifies that the TorchScript model produces the same output as the original.
    """
    test_input = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        original_output = original_model(test_input)
        scripted_output = scripted_model(test_input)

    # Check if outputs are close
    max_diff = torch.max(torch.abs(original_output - scripted_output)).item()
    print("\nVerification:")
    print(f"  Max difference between outputs: {max_diff:.6e}")

    if max_diff < 1e-5:
        print("  ✓ Outputs match! Conversion successful.")
        return True
    else:
        print("  ⚠ Outputs differ. Check conversion carefully.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViT model to TorchScript",
        formatter_class=RichHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="number of output classes (must match training)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vit_model_scripted.pt",
        help="output path for TorchScript model (.pt file)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="input image size (default 224)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="trace",
        choices=["trace", "script"],
        help="conversion method: trace (recommended) or script",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify conversion by comparing outputs",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load original model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_vit_model(args.num_classes, device, args.checkpoint)
    print(f"Model loaded with {args.num_classes} classes")

    # Convert to TorchScript
    print(f"\nConverting to TorchScript using {args.method} method...")
    scripted_model = convert_to_torchscript(
        model, device, img_size=args.img_size, method=args.method
    )

    # Verify conversion if requested
    if args.verify:
        verify_torchscript_model(scripted_model, model, device, args.img_size)

    # Save TorchScript model
    print(f"\nSaving TorchScript model to {args.output}...")
    torch.jit.save(scripted_model, args.output)
    print("Done!")

    # Print usage info
    print("\nTo load this model:")
    print(f"  model = torch.jit.load('{args.output}')")
    print("  model.eval()")
    print(
        f"  output = model(input_tensor)  # input shape: [batch, 3, {args.img_size}, {args.img_size}]"
    )


if __name__ == "__main__":
    main()
