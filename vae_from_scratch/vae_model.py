"""
Variational Autoencoder (VAE) implementation for image compression and generation.

The VAE compresses images into a latent space and reconstructs them back.
Architecture: 3x512x512 images -> 4x64x64 latent features -> reconstructed images
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder for image generation and reconstruction"""

    def __init__(self, in_channels=3, latent_dim=4, image_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder: Image -> Latent features
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),    # 512 -> 256
            self._conv_block(64, 128),            # 256 -> 128
            self._conv_block(128, 256),           # 128 -> 64
        )

        # Latent space projections
        self.fc_mu = nn.Conv2d(256, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(256, latent_dim, 1)

        # Decoder: Latent features -> Image
        self.decoder_input = nn.Conv2d(latent_dim, 256, 1)
        self.decoder = nn.Sequential(
            self._deconv_block(256, 128),         # 64 -> 128
            self._deconv_block(128, 64),          # 128 -> 256
            self._deconv_block(64, in_channels),  # 256 -> 512
        )

        self.final_activation = nn.Tanh()  # Output range: [-1, 1]

    def _conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv2d -> BatchNorm -> LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _deconv_block(self, in_channels, out_channels):
        """Deconvolutional block: ConvTranspose2d -> BatchNorm -> LeakyReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def decode(self, z):
        """Decode latent code to reconstruction"""
        x = self.decoder_input(z)
        x = self.decoder(x)
        x = self.final_activation(x)
        return x

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """Forward pass: encode -> reparameterize -> decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, x, mu, logvar

    def sample(self, num_samples, device):
        """Generate samples from prior distribution"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, 64, 64).to(device)
            samples = self.decode(z)
        return samples