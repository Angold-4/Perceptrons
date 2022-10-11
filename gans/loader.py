import torch.utils.data
from torchvision import transforms, datasets


def mnist_data():
    """Normalize the data into [-1, 1] space"""
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
        ])
    outdir = './dataset'
    return datasets.MNIST(root=outdir, train=True, transform=compose, download=True)

if __name__ == "__main__":
    data = mnist_data()

    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    print(len(data_loader))
