import torch
import re

from model import SimpleTransformer


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class TransformerClassifier:
    """Transformer text classifier for inference"""
    
    def __init__(self, model_path='transformer_model.pth'):
        self.device = get_device()
        self.model, self.vocab, self.config = self._load_model(model_path)
        self.class_names = {
            0: "Technology", 1: "Sports", 2: "Science", 3: "Music", 4: "Food"
        }
    
    def _load_model(self, model_path):
        """Load trained model and configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        vocab = checkpoint['vocab']
        config = checkpoint['model_config']
        
        model = SimpleTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            num_classes=config['num_classes'],
            max_length=config['max_length']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, vocab, config
    
    def _tokenize_text(self, text):
        """Tokenize text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def _encode_text(self, text):
        """Encode text to tensor"""
        tokens = self._tokenize_text(text)[:self.config['max_length']]
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad to max_length
        if len(encoded) < self.config['max_length']:
            encoded.extend([self.vocab['<PAD>']] * (self.config['max_length'] - len(encoded)))
        
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    
    def predict(self, text):
        """Predict class of input text"""
        encoded_text = self._encode_text(text).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(encoded_text)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_name = self.class_names.get(predicted_class, f"Class {predicted_class}")
        
        return {
            'class_id': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }

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
    
    try:
        classifier = TransformerClassifier()
        print(f"Testing Transformer Classifier on device: {classifier.device}")
        print("=" * 60)
        
        for i, text in enumerate(sample_texts):
            result = classifier.predict(text)
            
            print(f"\nText {i+1}: {text}")
            print(f"Predicted: {result['class_name']} (Class {result['class_id']})")
            print(f"Confidence: {result['confidence']:.2%}")
    
    except Exception as e:
        print(f"Error during testing: {e}")


def interactive_mode():
    """Interactive mode for testing custom text"""
    try:
        classifier = TransformerClassifier()
        print("\nInteractive Mode - Enter text to classify (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            result = classifier.predict(text)
            print(f"Predicted: {result['class_name']} (Class {result['class_id']})")
            print(f"Confidence: {result['confidence']:.2%}")
    
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
        print(f"Error: {e}")
