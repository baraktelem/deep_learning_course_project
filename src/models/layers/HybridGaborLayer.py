import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridGaborLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, gabor_kernel_size=7, ratio=0.5, pad_mode='constant',stride=1):
        super().__init__()

        self.n_param = int(out_channels * ratio)
        self.n_std = out_channels - self.n_param
        self.pad_mode = pad_mode
        self.stride = stride

        n_scales = int(math.ceil(math.sqrt(self.n_param))) # e.g. 4 scales
        n_angles = int(math.ceil(self.n_param / n_scales)) # e.g. 8 angles
        mesh_scales, mesh_angles = torch.meshgrid(
            torch.linspace(0, n_scales - 1, n_scales),
            torch.linspace(0, n_angles - 1, n_angles),
            indexing='ij'
        )
        mesh_scales = mesh_scales.flatten()[:self.n_param]
        mesh_angles = mesh_angles.flatten()[:self.n_param]
        min_wavelength = 3.0 # Smallest feature to look for
        base_wavelengths = min_wavelength * (2.0 ** (mesh_scales * 0.5))


        self.log_lambda = nn.Parameter(
            torch.log(base_wavelengths.unsqueeze(1).repeat(1, in_channels)) 
            + (torch.randn(self.n_param, in_channels) * 0.05) # Small jitter
        )


        base_angles = mesh_angles * (torch.pi / n_angles)
        self.theta = nn.Parameter(
            base_angles.unsqueeze(1).repeat(1, in_channels)
            + (torch.randn(self.n_param, in_channels) * 0.05)
        )

        sigma_factor = 0.8 
        base_sigma = base_wavelengths * sigma_factor
        
        self.log_sigma = nn.Parameter(
            torch.log(base_sigma.unsqueeze(1).repeat(1, in_channels))
            + (torch.randn(self.n_param, in_channels) * 0.05)
        )

        self.std_conv = nn.Conv2d(in_channels, self.n_std, conv_kernel_size, padding=conv_kernel_size//2, stride=stride, bias=False)

        with torch.no_grad():
            angles = torch.linspace(0, torch.pi, self.n_param).unsqueeze(1)
            self.theta.data = angles.repeat(1, in_channels) + (torch.randn_like(self.theta) * 0.1)

        self.conv_kernel_size = conv_kernel_size
        self.gabor_kernel_size = gabor_kernel_size
        self.in_channels = in_channels

    def generate_filters_quadrature(self, max_size):
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

    def forward(self, x):
        out_std = self.std_conv(x)

        f_real, f_imag = self.generate_filters_quadrature(x.size(3)-1)
        
        pad_amount = self.actual_kernel_size // 2
        x_padded = F.pad(x, (pad_amount, pad_amount, pad_amount, pad_amount), mode=self.pad_mode)
        
        out_real = F.conv2d(x_padded, f_real, stride=self.stride, padding=0)
        out_imag = F.conv2d(x_padded, f_imag, stride=self.stride, padding=0)
        
        out_param = torch.hypot(out_real, out_imag)
        out_param = F.avg_pool2d(out_param, kernel_size=3, stride=1, padding=1)
        
        return torch.cat([out_std, out_param], dim=1)