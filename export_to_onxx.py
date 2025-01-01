import torch
import os

def export_model_to_onnx(
    traced_model_path="traced_models/resnet50_imagenet_1k_model.pt", 
    output_path="onxx_models/resnet50_imagenet_1k_model.onnx"
):
    # Ensure the lambda-onnx directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Loading traced model...")
    model = torch.jit.load(traced_model_path)
    model.eval()

    # Create dummy input tensor matching your model's expected input size (1, 3, 160, 160)
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}},
    )
    print(f"Model exported successfully to {output_path}")

if __name__ == "__main__":
    export_model_to_onnx() 
    