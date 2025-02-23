import torch
from torchvision import transforms
from PIL import Image
from alexnet import AlexNet

class AlexNetInference:
    def __init__(self, model_path, class_labels_path=None):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = AlexNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load class labels if provided
        self.class_labels = []
        if class_labels_path:
            with open(class_labels_path, 'r') as f:
                self.class_labels = [line.strip() for line in f.readlines()]
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=5):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top k predictions
            top_prob, top_class = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                class_idx = top_class[0][i].item()
                if self.class_labels:
                    class_name = self.class_labels[class_idx]
                else:
                    class_name = f"Class {class_idx}"
                probability = top_prob[0][i].item()
                results.append((class_name, probability))
            
            return results

def main():
    # Example usage
    model_path = 'alexnet_model.pth'
    class_labels_path = 'path/to/class_labels.txt'  # Optional
    
    # Initialize inference
    inference = AlexNetInference(model_path, class_labels_path)
    
    # Make prediction
    image_path = 'path/to/test/image.jpg'
    predictions = inference.predict(image_path)
    
    # Print results
    print("\nTop 5 predictions:")
    for class_name, probability in predictions:
        print(f"{class_name}: {probability:.4f}")

if __name__ == "__main__":
    main() 