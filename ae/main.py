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
from net import Autoencoder, Encoder, Decoder, device
from torch.autograd.variable import Variable
import torchvision.utils as vutils
from IPython import display
from torch import randn, optim, nn, ones, zeros

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
    Autoencoder_model = Autoencoder(latent_dims=2).to(device)
    Autoencoder_model.load_state_dict(torch.load("./model/19ae.pth"))
    # Autoencoder_model.load_state_dict(torch.load("./model/finalae.pth"))

    sample = samples(16) # random

    datas = data.DataLoader(
            datasets.MNIST('./data',
                   transform=torchvision.transforms.ToTensor(),
                   download=True),
            batch_size=128,
            shuffle=True)

    plot_latent(Autoencoder_model, datas, num_batches=100)

    # display the sample img
    save_vecimg(sample, "sample.png")

    gen_vec = Autoencoder_model(sample)
    # display the out img
    save_vecimg(gen_vec, "out.png")


if __name__ == "__main__":
    main()
