import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time conditioning"""
    
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t):
        h = self.block1(x)
        h += self.time_mlp(t)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """Self-attention for spatial features"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class UNet(nn.Module):
    """U-Net architecture for diffusion models"""
    
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.init_conv = nn.Conv2d(c_in, 64, 3, padding=1)
        
        # Encoder (downsampling path)
        self.down1 = nn.Sequential(
            ResBlock(64, 128, time_dim),
            ResBlock(128, 128, time_dim),
            nn.Conv2d(128, 128, 4, 2, 1),
        )
        self.down2 = nn.Sequential(
            ResBlock(128, 256, time_dim),
            ResBlock(256, 256, time_dim),
            nn.Conv2d(256, 256, 4, 2, 1),
        )
        self.down3 = nn.Sequential(
            ResBlock(256, 256, time_dim),
            ResBlock(256, 256, time_dim),
            nn.Conv2d(256, 256, 4, 2, 1),
        )

        # Bottleneck
        self.bot1 = ResBlock(256, 512, time_dim)
        self.bot2 = ResBlock(512, 512, time_dim)
        self.bot3 = ResBlock(512, 256, time_dim)

        # Decoder (upsampling path)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            ResBlock(256, 256, time_dim),
            ResBlock(256, 256, time_dim),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            ResBlock(128, 256, time_dim),
            ResBlock(256, 128, time_dim),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            ResBlock(64, 128, time_dim),
            ResBlock(128, 64, time_dim),
        )
        
        # Final output
        self.output = nn.Conv2d(128, c_out, 1)

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial conv
        x1 = self.init_conv(x)
        
        # Encoder
        x2 = self.down1[0](x1, t)
        x2 = self.down1[1](x2, t)
        x2 = self.down1[2](x2)
        
        x3 = self.down2[0](x2, t)
        x3 = self.down2[1](x3, t)
        x3 = self.down2[2](x3)
        
        x4 = self.down3[0](x3, t)
        x4 = self.down3[1](x4, t)
        x4 = self.down3[2](x4)

        # Bottleneck
        x4 = self.bot1(x4, t)
        x4 = self.bot2(x4, t)
        x4 = self.bot3(x4, t)

        # Decoder with skip connections
        x = self.up1[0](x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up1[1](x, t)
        x = self.up1[2](x, t)
        
        x = self.up2[0](x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2[1](x, t)
        x = self.up2[2](x, t)
        
        x = self.up3[0](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3[1](x, t)
        x = self.up3[2](x, t)

        return self.output(x)


class NoiseScheduler:
    """Noise scheduler for diffusion process"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Useful precomputed values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_start, t, noise=None):
        """Add noise to clean images according to timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a, t, x_shape):
        """Extract values from tensor a at indices t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def sample_ddpm(model, noise_scheduler, shape, num_timesteps=1000, device="cpu"):
    """Sample from diffusion model using DDPM sampling"""
    model.eval()
    
    # Start with random noise
    img = torch.randn(shape, device=device)
    
    for i in reversed(range(num_timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)
        
        # Model prediction
        predicted_noise = model(img, t)
        
        # DDPM sampling step
        alpha = noise_scheduler.alphas[i]
        alpha_cumprod = noise_scheduler.alphas_cumprod[i] 
        beta = noise_scheduler.betas[i]
        
        img = (1 / alpha.sqrt()) * (img - beta / (1 - alpha_cumprod).sqrt() * predicted_noise)
        if i > 0:
            img += (beta.sqrt()) * noise
    
    model.train()
    return img


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule for improved sampling"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)