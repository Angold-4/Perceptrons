import pickle
import mnist_loader
from image2minst import *
import numpy as np
import cv2

with open('recongnition_model.pkl', 'rb') as f:
    net = pickle.load(f)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n_test = len(test_data)

test_img = cv2.imread("sample/7.png", cv2.IMREAD_GRAYSCALE)

sam_array = np.array(test_img).reshape((784, 1))

new_array = []

for i in range(len(sam_array)):
    new_array.append(sam_array[i,0]/255)

input_sample = np.array(new_array).reshape((784, 1))

print(input_sample)

test_results = np.argmax(net.feedforward(input_sample))
results = net.feedforward(input_sample)

print(test_results)
print(results)


"""
test_results = np.argmax(net.feedforward(sam_array))
results = net.feedforward(sam_array)

print(test_results)
print(results)

"""
"""
print(array)

sample_array = np.array(array).reshape((784, 1))

test_results = np.argmax(net.feedforward(sample_array))
results = net.feedforward(sample_array)

pixels = sample_array.reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.savefig('foo.png')

print(test_results)
print(results)
"""

print("{0} / {1}".format(net.evaluate(test_data), n_test))
