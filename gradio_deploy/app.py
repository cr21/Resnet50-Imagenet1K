import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from gradio.flagging import SimpleCSVLogger
from utils import GradioConfig

class Resnet50Imagenet1kGradioApp:
    def __init__(self,cfg: GradioConfig):
        self.device = cfg.device # Change this to 'cuda' if you have a GPU available
        
        # Validate model path parameters
            
        # Convert to strings if needed and create path
        model_dir = str(cfg.model_dir)
        model_file = str(cfg.model_file_name)
        model_full_path = Path(model_dir) / model_file
        
        # Verify the file exists
        if not model_full_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_full_path}")
            
        # load traced model
        self.model = torch.jit.load(model_full_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the same transforms used during training/testing
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.labels = cfg.labels

    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess image
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probs, indices = torch.topk(probabilities, k=5)
        print(f"Top 5 predictions:")
        for idx, prob in zip(indices, probs):
            print(f"idx: {idx}, label : {self.labels[idx]} , prob: {prob.item() * 100:.2f}%")  # Format probability to 2 decimal places)
        return {
            self.labels[idx]: float(prob)
            for idx, prob in zip(indices, probs)
        }

# Create classifier instance
cfg = GradioConfig()
classifier = Resnet50Imagenet1kGradioApp(cfg)


# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Resnet50 Imagenet 1k classifier",
    description="Upload an image to classify  Images",
    flagging_mode="never",
    flagging_callback=SimpleCSVLogger(),
    examples=["examples/blue_lobster.jpeg",
              "examples/lobster.jpeg",
              "examples/lobster2.jpeg",
              "examples/turtle.jpeg"]
)


if __name__ == "__main__":
    demo.launch() 