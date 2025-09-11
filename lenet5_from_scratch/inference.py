import torch
from torchvision import transforms
from PIL import Image
from model import LeNet5
import os

def predict_image(image_path):
    # Load the trained model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = LeNet5().to(device)
    model.load_state_dict(torch.load('lenet5_model.pth'))
    model.eval()

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

def test_all_digits():
    """Test the model on all generated test digits."""
    test_dir = 'test_digits'
    if not os.path.exists(test_dir):
        print("Please run generate_test_digits.py first to create test images.")
        return

    for digit in range(10):
        image_path = f'{test_dir}/digit_{digit}.png'
        if os.path.exists(image_path):
            prediction = predict_image(image_path)
            print(f'Digit {digit}: Predicted as {prediction} {"✓" if prediction == digit else "✗"}')

if __name__ == '__main__':
    test_all_digits()