import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.utils
import torch.distributions
import torchvision
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from net import VAE
from torch.autograd.variable import Variable
import torchvision.utils as vutils
from IPython import display
from torch import randn, optim, nn, ones, zeros
from numpy import linspace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_reconstructed(autoencoder, r0=(-10, 10), r1=(-15, 5), n=20):
    w = 28
    img = np.zeros((n*w, n*w))

    for i, y in enumerate(linspace(*r1, n)):
        for j, x in enumerate(linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat

    plt.imshow(img, extent=[*r0, *r1])
    plt.savefig('reconstructed.png')
    plt.close() 

def img2vec(image):
    return image.view(image.size(0), 784)

def vec2img(vector):
    return vector.view(vector.size(0), 1, 28, 28)

def samples(size):
    datas = data.DataLoader(
            datasets.MNIST('./data',
                   transform=torchvision.transforms.ToTensor(),
                   download=True),
            batch_size=size,
            shuffle=True)

    for d, _  in datas:
        # only choose the first one
        n = Variable(img2vec(d))
        return n;

def save_vecimg(vec, name):
    gen_img = vec2img(vec)
    images = gen_img.data.cpu()

    # Make horizontal grid from image tensor
    horizontal_grid = vutils.make_grid(
        images, normalize=True, scale_each=True)

    nrows = int(np.sqrt(16))
    grid = vutils.make_grid(
        images, nrow=nrows, normalize=True, scale_each=True)

    # Plot and save horizontal
    # fig = plt.figure(figsize=(16, 16))
    fig = plt.figure()
    plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    plt.axis('off') 
    display.display(plt.gcf())
    fig.savefig(name) # ignore this error if you are using pyright
    plt.close()


def plot_latent(autoencoder, data, num_batches=100):
    """show the plot space"""
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            plt.savefig('latent.png')
            plt.close() 
            break



def main():
    VAE_model = VAE().to(device)
    VAE_model.load_state_dict(torch.load("./model/finalvae.pth"))
    # Autoencoder_model.load_state_dict(torch.load("./model/finalae.pth"))

    sample = samples(16) # random

    datas = data.DataLoader(
            datasets.MNIST('./data',
                   transform=torchvision.transforms.ToTensor(),
                   download=True),
            batch_size=128,
            shuffle=True)

    # plot_latent(VAE_model, datas, num_batches=100)

    # display the sample img
    save_vecimg(sample, "sample.png")

    gen_vec = VAE_model(sample)
    # display the out img
    save_vecimg(gen_vec[0], "out.png")

    # plot_reconstructed(VAE_model)

if __name__ == "__main__":
    main()
