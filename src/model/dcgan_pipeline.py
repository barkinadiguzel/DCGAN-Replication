import torch
from torch import optim
from src.config import device, latent_dim, lr, beta1, num_epochs, batch_size
from src.backbone.generator_dcgan import Generator
from src.backbone.discriminator_dcgan import Discriminator
from src.loss.gan_loss import discriminator_loss, generator_loss

class DCGANPipeline:
    def __init__(self):
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))

    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = self.G(z)
      
        self.D.zero_grad()
        D_real = self.D(real_images)
        D_fake = self.D(fake_images.detach())
        loss_D = discriminator_loss(D_real, D_fake)
        loss_D.backward()
        self.opt_D.step()

        self.G.zero_grad()
        D_fake = self.D(fake_images)
        loss_G = generator_loss(D_fake)
        loss_G.backward()
        self.opt_G.step()

        return loss_D.item(), loss_G.item()
