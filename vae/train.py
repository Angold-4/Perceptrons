import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from net import VAE

image_size = 784
num_epochs = 15
batch_size = 128
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torchvision.datasets.MNIST(root='./',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

data_loader = data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

def main():
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device).view(-1, image_size)
            x_reconst, mu, log_var = model(x)
            
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            loss = reconst_loss + kl_div
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
    torch.save(model.state_dict(), "./model/finalvae.pth")

if __name__ == "__main__":
    main()
