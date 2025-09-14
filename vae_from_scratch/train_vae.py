import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datasets import load_dataset

from vae_model import VAE


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_transforms(image_size=256):
    """Create data preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] range
    ])


def transform_batch(examples, image_size=256):
    """Transform function for dataset"""
    preprocess = create_transforms(image_size)
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence
    
    Args:
        recon_x: Reconstructed images
        x: Original images  
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL term (beta-VAE)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_vae(num_epochs=50, batch_size=8, learning_rate=1e-3, image_size=256, latent_dim=4):
    """Train VAE on Pokemon dataset"""
    device = get_device()
    print(f"Training VAE on device: {device}")
    
    # Load and prepare dataset
    print("Loading Pokemon dataset...")
    try:
        dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
        dataset = dataset.with_transform(lambda x: transform_batch(x, image_size))
        dataset = dataset.remove_columns(['text', 'text_zh'])
        
        # Use subset for faster training
        dataset = dataset.shuffle(seed=42).select(range(min(800, len(dataset))))
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please install datasets: pip install datasets")
        return
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model and optimizer
    vae = VAE(in_channels=3, latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Create results directory
    os.makedirs("vae_results", exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        vae.train()
        train_loss = train_recon_loss = train_kl_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_images, _, mu, logvar = vae(images)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
        
        # Validation
        vae.eval()
        val_loss = val_recon_loss = val_kl_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                recon_images, _, mu, logvar = vae(images)
                
                loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar)
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1:3d}/{num_epochs}] '
              f'Train Loss: {train_loss:.0f} | Val Loss: {val_loss:.0f} | '
              f'Recon: {train_recon_loss/len(train_loader.dataset):.0f} | '
              f'KL: {train_kl_loss/len(train_loader.dataset):.1f}')
        
        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            _save_samples(vae, val_loader, device, epoch + 1, image_size)
    
    # Save final model and plots
    torch.save(vae.state_dict(), 'vae_model.pth')
    _plot_losses(train_losses, val_losses)
    
    print("Training completed!")
    print("Model saved as 'vae_model.pth'")
    print("Results saved in 'vae_results/' directory")


def _save_samples(vae, val_loader, device, epoch, image_size):
    """Save reconstruction and generation samples"""
    vae.eval()
    with torch.no_grad():
        # Get a validation batch for reconstruction
        batch = next(iter(val_loader))
        real_images = batch['images'][:8].to(device)
        
        # Reconstruct images
        recon_images, _, mu, logvar = vae(real_images)
        
        # Generate new samples
        z = torch.randn(8, vae.latent_dim, image_size//8, image_size//8).to(device)
        generated_images = vae.decode(z)
        
        # Save reconstructions
        comparison = torch.cat([real_images[:4], recon_images[:4]])
        save_image(comparison.clamp(-1, 1), f'vae_results/reconstruction_{epoch}.png', 
                  nrow=4, normalize=True, value_range=(-1, 1))
        
        # Save generations
        save_image(generated_images.clamp(-1, 1), f'vae_results/generated_{epoch}.png',
                  nrow=4, normalize=True, value_range=(-1, 1))


def _plot_losses(train_losses, val_losses):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('VAE Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_results/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    train_vae()