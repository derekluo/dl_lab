import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import CLIP, CLIPLoss, SimpleTokenizer, create_clip_model


class ImageTextDataset(Dataset):
    """Dataset for image-text pairs"""

    def __init__(self, data_file, image_root, transform=None, max_text_length=77):
        """
        Args:
            data_file: JSON file with image-text pairs
            image_root: Root directory for images
            transform: Image transformations
            max_text_length: Maximum text sequence length
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.transform = transform
        self.tokenizer = SimpleTokenizer(max_length=max_text_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = os.path.join(self.image_root, item['image'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Tokenize text
        text = item['caption']
        text_tokens = self.tokenizer.encode(text).squeeze(0)

        return image, text_tokens, text


def create_synthetic_dataset(num_samples=1000, save_path='data'):
    """Create a synthetic dataset for demonstration"""
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)

    # Generate simple synthetic images and captions
    data = []
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['circle', 'square', 'triangle']

    for i in range(num_samples):
        # Create simple colored shape
        color = np.random.choice(colors)
        shape = np.random.choice(shapes)

        # Create image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White background

        # Add colored shape
        if color == 'red':
            rgb = [255, 0, 0]
        elif color == 'blue':
            rgb = [0, 0, 255]
        elif color == 'green':
            rgb = [0, 255, 0]
        elif color == 'yellow':
            rgb = [255, 255, 0]
        elif color == 'purple':
            rgb = [128, 0, 128]
        else:  # orange
            rgb = [255, 165, 0]

        # Draw shape
        center_x, center_y = 112, 112
        size = 50

        if shape == 'circle':
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= size**2
        elif shape == 'square':
            mask = np.zeros((224, 224), dtype=bool)
            mask[center_y-size:center_y+size, center_x-size:center_x+size] = True
        else:  # triangle
            mask = np.zeros((224, 224), dtype=bool)
            for y in range(center_y-size, center_y+size):
                width = int(size * (1 - abs(y - center_y) / size))
                mask[y, center_x-width:center_x+width] = True

        img[mask] = rgb

        # Save image
        image_name = f'image_{i:04d}.png'
        Image.fromarray(img).save(os.path.join(save_path, 'images', image_name))

        # Create caption
        caption = f"a {color} {shape}"

        data.append({
            'image': image_name,
            'caption': caption
        })

    # Save dataset metadata
    with open(os.path.join(save_path, 'dataset.json'), 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created synthetic dataset with {num_samples} samples in {save_path}")


def get_transforms():
    """Get image preprocessing transforms"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    with tqdm(dataloader, desc=f'Epoch {epoch}') as pbar:
        for images, texts, _ in pbar:
            images = images.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            loss = criterion(logits_per_image, logits_per_text)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    # Metrics
    correct_i2t = 0
    correct_t2i = 0
    total_samples = 0

    with torch.no_grad():
        for images, texts, _ in dataloader:
            images = images.to(device)
            texts = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            loss = criterion(logits_per_image, logits_per_text)
            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            batch_size = images.shape[0]

            # Top-1 accuracy for image-to-text
            pred_i2t = torch.argmax(logits_per_image, dim=1)
            correct_i2t += (pred_i2t == torch.arange(batch_size, device=device)).sum().item()

            # Top-1 accuracy for text-to-image
            pred_t2i = torch.argmax(logits_per_text, dim=1)
            correct_t2i += (pred_t2i == torch.arange(batch_size, device=device)).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / num_batches
    acc_i2t = correct_i2t / total_samples
    acc_t2i = correct_t2i / total_samples

    return avg_loss, acc_i2t, acc_t2i


def plot_training_curves(train_losses, val_losses, val_accuracies, save_path='training_results.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    if val_accuracies:
        acc_i2t = [acc[0] for acc in val_accuracies]
        acc_t2i = [acc[1] for acc in val_accuracies]
        ax2.plot(epochs, acc_i2t, 'g-', label='Image-to-Text Acc')
        ax2.plot(epochs, acc_t2i, 'orange', label='Text-to-Image Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train CLIP model from scratch')
    parser.add_argument('--data_file', type=str, default='data/dataset.json',
                      help='Path to dataset JSON file')
    parser.add_argument('--image_root', type=str, default='data/images',
                      help='Root directory for images')
    parser.add_argument('--model_size', type=str, default='base',
                      choices=['tiny', 'small', 'base', 'large'],
                      help='Model size')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--save_path', type=str, default='clip_model.pth',
                      help='Path to save trained model')
    parser.add_argument('--create_synthetic', action='store_true',
                      help='Create synthetic dataset if data file not found')

    args = parser.parse_args()

    # Check if data file exists, create synthetic if needed
    if not os.path.exists(args.data_file) and args.create_synthetic:
        print("Data file not found. Creating synthetic dataset...")
        create_synthetic_dataset(num_samples=1000, save_path=os.path.dirname(args.data_file) or 'data')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create model
    model = create_clip_model(args.model_size)
    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Create dataset and dataloader
    transform = get_transforms()
    dataset = ImageTextDataset(args.data_file, args.image_root, transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # Loss and optimizer
    criterion = CLIPLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss, acc_i2t, acc_t2i = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append((acc_i2t, acc_t2i))

        # Step scheduler
        scheduler.step()

        # Print results
        print(f'Epoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  I2T Accuracy: {acc_i2t:.4f}')
        print(f'  T2I Accuracy: {acc_t2i:.4f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_i2t': acc_i2t,
                'val_acc_t2i': acc_t2i,
                'model_size': args.model_size,
            }, args.save_path)
            print(f'  Saved best model (val_loss: {val_loss:.4f})')

        print('-' * 50)

        # Plot training curves every 10 epochs
        if epoch % 10 == 0:
            plot_training_curves(train_losses, val_losses, val_accuracies)

    # Final training curves
    plot_training_curves(train_losses, val_losses, val_accuracies)
    print("Training completed!")
    print(f"Best model saved to: {args.save_path}")


if __name__ == '__main__':
    main()
