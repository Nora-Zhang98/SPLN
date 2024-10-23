import torch
import numpy as np
from torch import nn
from maskrcnn_benchmark.modeling.utils import cat

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class VAE_Encoder(nn.Module):
    def __init__(self, latent_size, layer_sizes_0):
        super(VAE_Encoder, self).__init__()
        self.fc1=nn.Linear(layer_sizes_0, layer_sizes_0)
        self.fc2 = nn.Linear(layer_sizes_0, int(layer_sizes_0/2))
        self.fc3 = nn.Linear(int(layer_sizes_0/2), latent_size*2)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class VAE_Decoder(nn.Module):
    def __init__(self, latent_size, layer_size_0):
        super(VAE_Decoder,self).__init__()
        input_size = latent_size
        self.fc0 = nn.Linear(input_size, latent_size*2)
        self.fc1 = nn.Linear(latent_size*2, int(layer_size_0/2))
        self.fc2 = nn.Linear(int(layer_size_0/2), layer_size_0)
        self.fc3 = nn.Linear(layer_size_0, layer_size_0)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.apply(weights_init)

    def forward(self, z):
        x = self.lrelu(self.fc0(z))
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x

class VAE_MODEL(nn.Module):
    def __init__(self, latent_size, layer_sizes_0, num_classes):
        super(VAE_MODEL, self).__init__()
        self.encoder = VAE_Encoder(latent_size, layer_sizes_0)
        self.decoder = VAE_Decoder(latent_size, layer_sizes_0)
        self.num_classes = num_classes
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.apply(weights_init)

    def forward(self, feats):
        mu, log_var = self.encoder(feats)
        z_embedding = self.reparameterize(mu, log_var)
        De_z_embedding = self.decoder(z_embedding)
        return [De_z_embedding, mu, log_var]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

def VAE_loss(input_features, rec_features, mu, log_var):
    L_Re = torch.mean(torch.linalg.norm((input_features - rec_features), dim=1))
    L_kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    L_vae = L_Re + L_kl
    return L_vae