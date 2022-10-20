from net import Autoencoder, Decoder, Encoder, device
from torch.utils import data
from torchvision import datasets
import torch
import torchvision

def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            # loss fn: direct calculate distance
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
        print("Epoch ", epoch, "/", epochs-1, " Finished.")

    torch.save(autoencoder.state_dict(), "./model/finalae.pth")

    return autoencoder

def main():
    latent_dims = 16
    autoencoder = Autoencoder(latent_dims).to(device)

    datas = data.DataLoader(
            datasets.MNIST('./data',
                   transform=torchvision.transforms.ToTensor(),
                   download=True),
            batch_size=128,
            shuffle=True)

    train(autoencoder, datas)

if __name__ == "__main__":
    main()
