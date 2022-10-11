import network
import mnist_loader
import pickle

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

with open('recongnition_model.pkl', 'wb') as f:
    pickle.dump(net, f)
