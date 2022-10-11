import torch
import train
import numpy as np
from network import GeneratorNet
from train import vec2img
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from IPython import display

generator_model = GeneratorNet()
generator_model.load_state_dict(torch.load("./model/199generator.pth"))

test_noise = train.noise(1) # random

gen_img = vec2img(generator_model(test_noise))
images = gen_img.data.cpu()

# images = gen_img.transpose(1,3)
# images = torch.from_numpy(gen_img)

# Make horizontal grid from image tensor
horizontal_grid = vutils.make_grid(
    images, normalize=True, scale_each=True)
# Make vertical grid from image tensor
nrows = int(np.sqrt(1))
grid = vutils.make_grid(
    images, nrow=nrows, normalize=True, scale_each=True)

# Plot and save horizontal
fig = plt.figure(figsize=(16, 16))
plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
plt.axis('off')
display.display(plt.gcf())
fig.savefig('out.png')
plt.close()

"""
# Save squared
fig = plt.figure()
plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
plt.axis('off')
self._save_images(fig, epoch, n_batch)
plt.close()
"""
