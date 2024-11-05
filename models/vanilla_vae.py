import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):
    
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        # Build Encoder, q_phi(z|x)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding= 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        # hidden_dims[-1]*4 = flatten된 크기 64x64x3 -> 2x2x512 -> 512 x 4 이렇게 되니까
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        
        # Build Decoder, p_theta(x|z)
        modules = []
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                    hidden_dims[i + 1],
                    kernel_size=3,
                    stride = 2,
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU())
            )
        
        
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Input을 받아 encoder network를 통해 인코딩 -> latent vector 반환

        Args:
            input (Tensor): Input tensor to encoder [B x C x H x W]

        Returns:
            List[Tensor]: List of latent codes
            mu: [B x latent_dim]
            log_var = [B x latent_dim]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        # Split the result into mu and var components of the Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        """
        주어진 latent codes를 image space로 매핑

        Args:
            z (Tensor): [B x D]

        Returns:
            Tensor: [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick을 통해 N(0, 1) 에서 sample -> N(mu, var)

        Args:
            mu (Tensor): Mean of the latent Gaussian [B x D]
            logvar (Tensor): Standard deviation of the latent Gaussian [B x D]

        Returns:
            Tensor: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    def loss_function(self, 
                      *args,
                      **kwargs) -> dict:
        """
        VAE loss 함수 계산
        KL(N(mu, sig), N(0, 1)) = log 1/sig + (sig^2 + mu^2)/2 - 1/2

        Returns:
            dict:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset, beta - VAE?
        recons_loss = F.mse_loss(recons, input)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        loss = recons_loss + kld_weight * kld_loss        
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': - kld_loss.detach()}
    
    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Latent space로부터 샘플링 후 그에 대응하는
        image space map 반환

        Args:
            num_samples (int): Number of samples
            current_device (int): Device to run the model

        Returns:
            Tensor:
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        
        z = z.to(current_device)
        
        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        주어진 input image x로부터 reconstructed image를 반환

        Args:
            x (Tensor): [B x C x H x W]

        Returns:
            Tensor: [B x C x H x W]
        """

        return self.forward(x)[0]