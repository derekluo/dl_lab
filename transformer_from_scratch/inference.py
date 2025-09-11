import torch
import re
from model import SimpleTransformer

def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load('transformer_model.pth', map_location=device)
    vocab = checkpoint['vocab']
    model_config = checkpoint['model_config']

    # Initialize model
    model = SimpleTransformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        num_classes=model_config['num_classes'],
        max_length=model_config['max_length']
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab, model_config, device

def tokenize_text(text):
    """Simple tokenization function"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens

def encode_text(text, vocab, max_length):
    """Encode text to tensor"""
    tokens = tokenize_text(text)[:max_length]
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    # Pad to max_length
    if len(encoded) < max_length:
        encoded.extend([vocab['<PAD>']] * (max_length - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def predict_text(text):
    """Predict the class of input text"""
    model, vocab, model_config, device = load_model_and_vocab()

    # Encode text
    encoded_text = encode_text(text, vocab, model_config['max_length']).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(encoded_text)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

def get_class_name(class_id):
    """Convert class ID to class name"""
    class_names = {
        0: "Technology",
        1: "Sports",
        2: "Science",
        3: "Music",
        4: "Food"
    }
    return class_names.get(class_id, f"Class {class_id}")

def test_sample_texts():
    """Test the model on sample texts"""
    sample_texts = [
        "This computer program uses advanced algorithms and data structures",
        "The basketball team won the championship game with a final score",
        "Scientists conducted experiments to test their hypothesis about the theory",
        "The musician played a beautiful melody on the piano during the concert",
        "The chef prepared a delicious meal with fresh ingredients from the market",
        "Programming languages like Python are used for software development",
        "The football match ended in overtime with an exciting goal",
        "Research shows that this method improves experimental results significantly"
    ]

    print("Testing Transformer Text Classifier")
    print("=" * 50)

    for i, text in enumerate(sample_texts):
        try:
            predicted_class, confidence = predict_text(text)
            class_name = get_class_name(predicted_class)

            print(f"\nText {i+1}: {text}")
            print(f"Predicted: {class_name} (Class {predicted_class})")
            print(f"Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"Error processing text {i+1}: {e}")

def interactive_mode():
    """Interactive mode for testing custom text"""
    print("\nInteractive Mode - Enter text to classify (type 'quit' to exit)")
    print("=" * 50)

    while True:
        text = input("\nEnter text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            print("Please enter some text.")
            continue

        try:
            predicted_class, confidence = predict_text(text)
            class_name = get_class_name(predicted_class)

            print(f"Predicted: {class_name} (Class {predicted_class})")
            print(f"Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    try:
        # Test on predefined samples
        test_sample_texts()

        # Interactive mode
        interactive_mode()

    except FileNotFoundError:
        print("Model file 'transformer_model.pth' not found.")
        print("Please run 'python train.py' first to train the model.")
    except Exception as e:
        print(f"Error loading model: {e}")
