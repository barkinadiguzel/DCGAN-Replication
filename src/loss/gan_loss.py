import torch
import torch.nn as nn

adversarial_loss = nn.BCELoss()

def discriminator_loss(D_real, D_fake):
    real_labels = torch.ones_like(D_real)
    fake_labels = torch.zeros_like(D_fake)
    real_loss = adversarial_loss(D_real, real_labels)
    fake_loss = adversarial_loss(D_fake, fake_labels)
    return real_loss + fake_loss

def generator_loss(D_fake):
    real_labels = torch.ones_like(D_fake)
    return adversarial_loss(D_fake, real_labels)
