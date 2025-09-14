import torch
import os
from torchvision import transforms
from PIL import Image

from model import LeNet5


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class LeNet5Predictor:
    """LeNet5 inference class for digit recognition"""
    
    def __init__(self, model_path='lenet5_model.pth'):
        self.device = get_device()
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def _load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = LeNet5().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def predict_image(self, image_path):
        """Predict digit from image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('L')  # Grayscale
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted].item()
        
        return predicted, confidence

def test_all_digits():
    """Test the model on all generated test digits"""
    test_dir = 'test_digits'
    if not os.path.exists(test_dir):
        print("Please run generate_test_digits.py first to create test images.")
        return
    
    try:
        predictor = LeNet5Predictor()
        print(f"Testing LeNet5 on device: {predictor.device}")
        print("=" * 40)
        
        correct = 0
        total = 0
        
        for digit in range(10):
            image_path = f'{test_dir}/digit_{digit}.png'
            if os.path.exists(image_path):
                prediction, confidence = predictor.predict_image(image_path)
                is_correct = prediction == digit
                correct += is_correct
                total += 1
                
                status = "✓" if is_correct else "✗"
                print(f'Digit {digit}: Predicted {prediction} (conf: {confidence:.2%}) {status}')
        
        accuracy = correct / total if total > 0 else 0
        print("=" * 40)
        print(f'Overall Accuracy: {accuracy:.2%} ({correct}/{total})')
        
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == '__main__':
    test_all_digits()