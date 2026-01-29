import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdditiveHybridGaborLayer(nn.Module):
    """
    A hybrid layer that uses Monogenic Difference-of-Gaussians (DoG) 
    to create an attention mechanism for subsequent Gabor filtering.
    """
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, gabor_kernel_size=7, ratio=1, pad_mode='constant',stride=1, norm_and_activation=False):
        super().__init__()

        self.n_param = int(out_channels * ratio)
        self.n_std = out_channels
        self.pad_mode = pad_mode
        self.stride = stride
        self.conv_kernel_size = conv_kernel_size
        self.gabor_kernel_size = gabor_kernel_size

        # The internal Gabor logic will handle double the channels due to the attention concatenation
        self.in_channels = in_channels * 2
        self.norm_and_activation = norm_and_activation

        # --- Gabor Parameter Grid (Frequency & Orientation) ---
        n_scales = int(math.ceil(math.sqrt(self.n_param)))
        n_angles = int(math.ceil(self.n_param / n_scales))
        mesh_scales, mesh_angles = torch.meshgrid(
            torch.linspace(0, n_scales - 1, n_scales),
            torch.linspace(0, n_angles - 1, n_angles),
            indexing='ij'
        )
        mesh_scales = mesh_scales.flatten()[:self.n_param]
        mesh_angles = mesh_angles.flatten()[:self.n_param]
        min_wavelength = 3.0
        base_wavelengths = min_wavelength * (2.0 ** (mesh_scales * 0.5))


        self.log_lambda = nn.Parameter(
            torch.log(base_wavelengths.unsqueeze(1).repeat(1, self.in_channels)) 
            + (torch.randn(self.n_param, self.in_channels) * 0.05) # Small jitter
        )


        base_angles = mesh_angles * (torch.pi / n_angles)
        self.theta = nn.Parameter(
            base_angles.unsqueeze(1).repeat(1, self.in_channels)
            + (torch.randn(self.n_param, self.in_channels) * 0.05)
        )

        sigma_factor = 0.8 
        base_sigma = base_wavelengths * sigma_factor
        
        self.log_sigma = nn.Parameter(
            torch.log(base_sigma.unsqueeze(1).repeat(1, self.in_channels))
            + (torch.randn(self.n_param, self.in_channels) * 0.05)
        )

        # Standard Conv branch
        self.std_conv = nn.Conv2d(in_channels, self.n_std, conv_kernel_size, padding=conv_kernel_size//2, stride=stride, bias=False)


        # --- Difference of Gaussians (DoG) Initialization ---
        # Used for blob detection and edge enhancement
        min_sigma = 1.0
        max_sigma = 5.0
        
        rand_scales = torch.rand(in_channels) * (max_sigma - min_sigma) + min_sigma
        steps = rand_scales
        
        sigma1_init = steps
        sigma2_init = steps * 1.6 + (torch.rand_like(steps) * 0.2 - 0.1)
        
        log_sigma1 = torch.log(sigma1_init - 0.05)
        log_sigma2 = torch.log(sigma2_init - 0.05)
        
        # log-space sigmas for the two Gaussian kernels in DoG
        self.log_dog_sigma1 = nn.Parameter(log_sigma1.view(in_channels, 1, 1, 1))
        self.log_dog_sigma2 = nn.Parameter(log_sigma2.view(in_channels, 1, 1, 1))

        if norm_and_activation:
            self.bn_std = nn.BatchNorm2d(out_channels)
            self.bn_param = nn.BatchNorm2d(out_channels)
            self.bn_final = nn.BatchNorm2d(out_channels)


    def generate_filters_quadrature(self, max_size):
        """Synthesizes Gabor kernels (Standard Real/Imaginary pair)."""
        K = min(self.gabor_kernel_size, max_size)
        self.actual_kernel_size = K
        r = K // 2
        y_grid, x_grid = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32, device=self.theta.device),
            torch.arange(-r, r + 1, dtype=torch.float32, device=self.theta.device),
            indexing='ij'
        )
        y_grid = y_grid.view(1,1, K, K)
        x_grid = x_grid.view(1,1, K, K)
        theta = self.theta.view(self.n_param, self.in_channels, 1, 1)
        sigma = torch.exp(self.log_sigma).view(self.n_param, self.in_channels, 1, 1) + 0.05
        psi = 0
        lambd = torch.exp(self.log_lambda).view(self.n_param, self.in_channels, 1, 1) + 0.05
        gamma = 1.0

        x_prime = x_grid * torch.cos(theta) + y_grid * torch.sin(theta)
        y_prime = -x_grid * torch.sin(theta) + y_grid * torch.cos(theta)
        gaussian = torch.exp(
            -(x_prime**2 + (gamma**2 * y_prime**2)) / (2 * sigma**2 +1e-5)
        )
        carrier_cos = torch.cos((2 * torch.pi * x_prime / (lambd + 1e-5)) + psi)
        carrier_sin = torch.sin((2 * torch.pi * x_prime / (lambd + 1e-5)) + psi)

        real_filter = gaussian * carrier_cos
        imag_filter = gaussian * carrier_sin
        
        if self.training:
             real_filter = real_filter + (torch.randn_like(real_filter) * 0.05)
             imag_filter = imag_filter + (torch.randn_like(imag_filter) * 0.05)

        return real_filter, imag_filter

    def generate_monogenic_dog_filters(self, max_size):
        """
        Synthesizes Monogenic DoG filters. 
        Returns:
        1. Real: The DoG (bandpass filter)
        2. Imag X/Y: The Riesz Transform of the DoG (gradient-like)
        """
        K = min(self.gabor_kernel_size, max_size)
        self.actual_kernel_size = K
        r = K // 2
        y, x = torch.meshgrid(
            torch.arange(-r, r + 1, device=self.log_dog_sigma1.device),
            torch.arange(-r, r + 1, device=self.log_dog_sigma1.device),
            indexing='ij'
        )

        y = y.view(1, 1, K, K)
        x = x.view(1, 1, K, K)
        r2 = x**2 + y**2

        s1 = torch.exp(self.log_dog_sigma1) + 0.05
        s2 = torch.exp(self.log_dog_sigma2) + 0.05

        norm1 = 1.0 / (2 * torch.pi * s1**2)
        norm2 = 1.0 / (2 * torch.pi * s2**2)
        
        # Standard Gaussian kernels
        g1_base = torch.exp(-r2 / (2 * s1**2)) * norm1
        g2_base = torch.exp(-r2 / (2 * s2**2)) * norm2

        # Real part is the Difference of Gaussians (Isotropic bandpass)
        real_filter = g1_base - g2_base

        # Imaginary parts are the spatial derivatives (Riesz transform)
        # These help determine the local phase and orientation
        g1_x = (-x / (s1**2)) * g1_base
        g1_y = (-y / (s1**2)) * g1_base
        
        g2_x = (-x / (s2**2)) * g2_base
        g2_y = (-y / (s2**2)) * g2_base

        imag_filter_x = g1_x - g2_x
        imag_filter_y = g1_y - g2_y

        if self.training:
            noise = torch.randn_like(real_filter) * 0.05
            return real_filter + noise, imag_filter_x + noise, imag_filter_y + noise

        return real_filter, imag_filter_x, imag_filter_y

    def forward(self, x):
        # 1. Standard convolution path
        out_std = self.std_conv(x)

        # stage 1: Monogenic Signal Analysis
        # Extract local energy regardless of orientation using DoG
        f_real, f_imag_x, f_imag_y = self.generate_monogenic_dog_filters(x.size(3)-1)
        
        pad_amount = self.actual_kernel_size // 2
        x_padded = F.pad(x, (pad_amount, pad_amount, pad_amount, pad_amount), mode=self.pad_mode)

        # Grouped convolution: each input channel gets its own learnable DoG filter
        out_real = F.conv2d(x_padded, f_real, stride=self.stride, padding=0, groups=self.in_channels//2)
        out_x    = F.conv2d(x_padded, f_imag_x, stride=self.stride, padding=0, groups=self.in_channels//2)
        out_y    = F.conv2d(x_padded, f_imag_y, stride=self.stride, padding=0, groups=self.in_channels//2)

        # Local Energy of the monogenic signal
        out_imag_mag = torch.sqrt(out_x**2 + out_y**2 + 1e-8)
        out_monogenic = torch.sqrt(out_real**2 + out_imag_mag**2 + 1e-8)

        out_dog_final = F.avg_pool2d(out_monogenic, kernel_size=3, stride=1, padding=1)

        # stage 2: Attention Masking
        # sigmoid maps energy to [0, 1]. High energy areas (edges/blobs) get highlighted.
        attn_mask = torch.sigmoid(out_dog_final)
        # Concatenate original input with its 'attended' version
        x = torch.cat([x, x * attn_mask], dim=1)
        
        # stage 3: Oriented Gabor Filtering
        f_real, f_imag = self.generate_filters_quadrature(x.size(3)-1)
        pad_amount = self.actual_kernel_size // 2
        x_padded = F.pad(x, (pad_amount, pad_amount, pad_amount, pad_amount), mode=self.pad_mode)
        
        out_real = F.conv2d(x_padded, f_real, stride=self.stride, padding=0)
        out_imag = F.conv2d(x_padded, f_imag, stride=self.stride, padding=0)

        # Final Gabor magnitude
        out_param = torch.sqrt(out_real**2 + out_imag**2 + 1e-8)
        out_param = F.avg_pool2d(out_param, kernel_size=3, stride=1, padding=1)

        if self.norm_and_activation:
            out_param = self.bn_param(out_param)
            out_std = self.bn_std(out_std)
            # Max-pooling across the two representations (standard vs gabor)
            return torch.max(out_std, out_param)
        
        return out_std, out_param
    