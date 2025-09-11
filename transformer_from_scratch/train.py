import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import re
import matplotlib.pyplot as plt
from model import SimpleTransformer

class TextDataset(Dataset):
    """Simple text dataset for demonstration"""
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Simple tokenization and encoding
        tokens = self.tokenize(text)
        encoded = self.encode_tokens(tokens)
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def tokenize(self, text):
        # Simple tokenization: lowercase and split on whitespace/punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens[:self.max_length]
    
    def encode_tokens(self, tokens):
        # Encode tokens to indices, pad to max_length
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(encoded) < self.max_length:
            encoded.extend([self.vocab['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
            
        return encoded

def build_vocab(texts, min_freq=2, max_vocab_size=5000):
    """Build vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        word_counts.update(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    most_common = word_counts.most_common(max_vocab_size - 2)
    for word, count in most_common:
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def create_sample_data(num_samples=2000, num_classes=5):
    """Create sample text data for demonstration"""
    texts = []
    labels = []
    
    # Simple patterns for different classes
    patterns = [
        ["computer", "software", "technology", "programming", "code", "algorithm", "data", "system"],
        ["sports", "football", "basketball", "game", "team", "player", "match", "score"],
        ["science", "research", "experiment", "theory", "study", "analysis", "discovery", "method"],
        ["music", "song", "artist", "album", "concert", "melody", "rhythm", "instrument"],
        ["food", "restaurant", "cooking", "recipe", "delicious", "taste", "ingredient", "meal"]
    ]
    
    for i in range(num_samples):
        class_idx = i % num_classes
        # Create a sentence using words from the pattern
        words = np.random.choice(patterns[class_idx], size=np.random.randint(8, 20))
        text = " ".join(words)
        texts.append(text)
        labels.append(class_idx)
    
    return texts, labels

def train_model(num_epochs=15, batch_size=32, learning_rate=0.001):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sample data
    print("Creating sample data...")
    texts, labels = create_sample_data(num_samples=2000, num_classes=5)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=1, max_vocab_size=5000)
    vocab_size = len(vocab)
    num_classes = len(set(labels))
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")
    
    # Create datasets and dataloaders
    max_length = 64
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_classes=num_classes,
        max_length=max_length
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Starting training...\n")
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save the model and vocabulary
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_config': {
            'vocab_size': vocab_size,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'num_classes': num_classes,
            'max_length': max_length
        }
    }, 'transformer_model.pth')
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed. Final accuracy: {test_accuracies[-1]:.2f}%")
    print("Model saved as 'transformer_model.pth'")
    print("Training results saved as 'training_results.png'")

if __name__ == '__main__':
    train_model()