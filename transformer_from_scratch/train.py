import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import SimpleTransformer


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class TextDataset(Dataset):
    """Simple text dataset for classification"""
    
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
        encoded = self._encode_text(text)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def _encode_text(self, text):
        # Tokenize and encode
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()[:self.max_length]
        
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad to max_length
        if len(encoded) < self.max_length:
            encoded.extend([self.vocab['<PAD>']] * (self.max_length - len(encoded)))
        
        return encoded

def build_vocab(texts, min_freq=2, max_vocab_size=5000):
    """Build vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        word_counts.update(clean_text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_vocab_size - 2):
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

def train_model(num_epochs=15, batch_size=32, learning_rate=0.001, max_length=64):
    device = get_device()
    print(f"Using device: {device}")

    # Create and split data
    print("Creating sample data...")
    texts, labels = create_sample_data(num_samples=2000, num_classes=5)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=1, max_vocab_size=5000)
    vocab_size = len(vocab)
    num_classes = len(set(labels))

    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_texts):,}")
    print(f"Test samples: {len(test_texts):,}")

    # Create datasets and data loaders
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_classes=num_classes,
        max_length=max_length
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...\n")

    # Training
    train_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluate
        accuracy = _evaluate_model(model, test_loader, device)
        test_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1:2d}/{num_epochs}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

    # Save results
    _save_results(model, vocab, train_losses, test_accuracies, vocab_size, num_classes, max_length)
    print(f"\nTraining completed! Final accuracy: {test_accuracies[-1]:.2f}%")


def _evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total


def _save_results(model, vocab, train_losses, test_accuracies, vocab_size, num_classes, max_length):
    """Save model and training plots"""
    # Save model
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

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Model saved as 'transformer_model.pth'")
    print("Training plots saved as 'training_results.png'")

if __name__ == '__main__':
    train_model()