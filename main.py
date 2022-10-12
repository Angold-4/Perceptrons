import torchvision.utils as vutils
from matplotlib import pyplot as plt
from IPython import display
import pickle
import numpy as np
import cv2
import torch
import sys
from torch import nn, randn
from torch.autograd.variable import Variable
from torchvision import transforms

def img2vec(image):
    return image.view(image.size(0), 784)

def vec2img(vector):
    return vector.view(vector.size(0), 1, 28, 28)

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(randn(size, 100))
    return n

class GeneratorNet(nn.Module):
    """
    A three hidden layer discriminative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__();
        n_features = 100; # latent variable vector
        n_out = 784;      # The photo generated

        self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LeakyReLU(0.2),
                )

        self.hidden1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                )

        self.hidden2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                )

        self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Tanh()  # [-1, 1]
                )

    def forward(self, x):
        x = self.hidden0(x);
        x = self.hidden1(x);
        x = self.hidden2(x);
        x = self.out(x);
        return x;

# total arguments
n = len(sys.argv)

if n != 2:
    print("Usage: python main.py <num>")
    exit()

ex = int(sys.argv[1]) # expected

# load the trained recongnizer model
with open('./model/recongnition_model.pkl', 'rb') as f:
    rec_net = pickle.load(f)

# load the trained generator model
generator_model = GeneratorNet()
generator_model.load_state_dict(torch.load("./model/199generator.pth"))

inv_normalize = transforms.ToTensor()

while True:
    rand = noise(1)

    gen_vec = generator_model(rand)
    images = gen_vec.data.cpu()


    # Make horizontal grid from image tensor
    horizontal_grid = vutils.make_grid(
        images, normalize=True, scale_each=True)

    feed_img = horizontal_grid[0].numpy().reshape((784, 1))

    results = rec_net.feedforward(feed_img)

    rec_value = np.argmax(results)
    p = float(results[rec_value])

    if p > 0.995 and rec_value == ex:
        images = vec2img(gen_vec).data.cpu()
        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=True, scale_each=True)

        # Plot and save horizontal
        fig = plt.figure()
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        display.display(plt.gcf())
        fig.savefig('out.png') # ignore this error if you are using pyright
        plt.close()
        break;
