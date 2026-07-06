import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import RobustAttentionGuidedEdgeViT

CLASSES = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = RobustAttentionGuidedEdgeViT(num_classes=7, edge_mode='sobel')
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Warning: Could not load weights. Using untrained model for UI demonstration. Error: {e}")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return None

    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs, _ = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction Confidence"),
    title="Attention-Guided Leaf Disease Classifier",
    description="Upload an image of a crop leaf to identify potential diseases. This model uses a Vision Transformer enhanced with edge-detection structural biases.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
