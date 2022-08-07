import network
import pickle
import mnist_loader

with open('recongnition_model.pkl', 'rb') as f:
    net = pickle.load(f)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n_test = len(test_data)
print("{0} / {1}".format(net.evaluate(test_data), n_test))
