import pickle
import sys
import numpy as np
import cv2

# load the trained model
with open('recongnition_model.pkl', 'rb') as f:
    net = pickle.load(f)

# total arguments
n = len(sys.argv)

if n != 2:
    print("Usage: python3 main.py <your image>")
    exit()

filename = sys.argv[1]

test_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

sam_array = np.array(test_img).reshape((784, 1))
new_array = []
for i in range(len(sam_array)):
    new_array.append(sam_array[i,0]/255)

input_sample = np.array(new_array).reshape((784, 1))
results = net.feedforward(input_sample)
test_results = np.argmax(results)

print("Recognized Value: {0}".format(test_results))
value = results[test_results] * 100
print("Confidence: " + format(value) + "%")

