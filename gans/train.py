from utils import Logger
import torch
from loader import mnist_data
from torch import randn, optim, nn, ones, zeros
import torch.utils.data
from network import GeneratorNet, DiscrimniatorNet
from torch.autograd.variable import Variable

generator = GeneratorNet();
discriminator = DiscrimniatorNet();

g_optimizer = optim.Adam(generator.parameters(), lr = 0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0002)

loss = nn.BCELoss()

"""
The helper functions
"""
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

def ones_target(size):
    data = Variable(ones(size, 1))
    return data;

def zeros_target(size):
    data = Variable(zeros(size, 1))
    return data;


"""
Actual train function
"""
def train_generator(optimizer, fake_data):
    N = fake_data.size(0)

    # reset gradients
    optimizer.zero_grad()

    # 1. sample noise to generate fake data
    prediction = discriminator(fake_data) # ge the prediction

    """
    Key fact: at the same time, the discriminator is getting better too!
    """

    # 2. calculate the error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward() # backpropagate

    # 3. Update weight with gradients
    optimizer.step()

    return error

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    # 1. train on real data
    prediction_real = discriminator(real_data)
    # calculate the error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 2. train on fake data
    prediction_fake = discriminator(fake_data)
    # calculate the error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 3. Update the weight with gradients
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def main():
    torch.set_num_threads(8)
    num_test_samples = 16
    test_noise = noise(num_test_samples) # predifined noise

    data = mnist_data()
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle = True) 
    num_batches = len(data_loader)

    logger = Logger(model_name = "VGAN", data_name = "MNIST")

    # Total number of epoch to train
    num_epochs = 200

    for epoch in range(num_epochs):
        for n_batch, (real_batch, _) in enumerate(data_loader):
            N = real_batch.size(0)

            # 1. Train the discriminator
            real_data = Variable(img2vec(real_batch))
            fake_data = generator(noise(N)).detach()
            # generate a new tensor using generator

            # Train D
            d_error, d_pred_real, d_pred_fake = \
                    train_discriminator(d_optimizer, real_data, fake_data)

            # 2. Train the generator
            fake_data = generator(noise(N))
            g_error = train_generator(g_optimizer, fake_data)

            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress every few batches
            if (n_batch) % 100 == 0: 
                test_images = vec2img(generator(test_noise))
                test_images = test_images.data.cpu()
                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                );
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

        torch.save(generator.state_dict(), "./model/" + str(epoch) + "generator.pth")
        torch.save(discriminator.state_dict(), "./model/" + str(epoch) + "discriminator.pth")

if __name__ == "__main__":
    main()
