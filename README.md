# Perceptrons
Using neural nets to recognize handwritten digits, follow the tutorial made by Michael Nielsen.

![digits](./photo/digits.png)

In each hemisphere of our brain, humans have a primary visual cortex, also known as V1, containing *140* million neurons, with tens of billions of connections between them. And yet human vision involves not just V1, but an entire series of visual cortices - V2, V3, V4, and V5 - doing progressively more complex image processing. **We carry in our heads a supercomputer, tuned by evolution over hundreds of millions of years, and superbly adapted to understand the visual world.**

In this project we implement a computer program that training a neural network which learns to recognize handwritten digits.

## Quick Start

### Prerequises

#### Installing via pip

```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

#### Installing via a package manager
##### Ubuntu and Debian
```
sudo apt-get install python3-numpy python3-scipy python3-matplotlib ipython3 python3-notebook python3-pandas python3-sympy
```

### 1. Train the neural network
If all things are ready, run
```
python3 train.py
```

Then you will see the different patches(epoch) of training will be complete. 

```
Epoch 0 complete
Epoch 1 complete
Epoch 2 complete
...
Epoch 28 complete
Epoch 29 complete
```

We use the [MNIST DB](http://yann.lecun.com/exdb/mnist/) as our training data, as you can see in [data/mnist.pkl.gz](data/mnist.pkl.gz).

Since we are using the **stochastic gradient descent** technique as our learning method, we divide the training data into 30 pieces(or batches). If you want to check the trained output each round, you can change the [train.py](./train.py) file a little bit by changing the `test_data=None` into `test_data=test_data` in line 9.

When you run that script again, you'll see the trained results of each epoch:

```
Epoch 0: 8980 / 10000
Epoch 1: 9162 / 10000
Epoch 2: 9229 / 10000
...
Epoch 29: 9438 / 10000
```

As you can see, after just a single epoch this has reached 8980 out of 10,000, and the number continues to grow, and the trained network gives us a classification rate of about 95 percent - *94.38* percent at its peak ("Epoch 29").

After the training complete, the program will save the trained model in [recongnition_model.pkl](./recongnition_model) for further use.

### 2. Using the trained model to recognize image

We also provide a way to recognize the image (png, jpeg) directly -- rather than using the pkl training data as one of the application of this small neural network.

You can use this application by calling:
```
python3 main.py <img filename>
```

For example: Consider some random sample images in [sample](./sample).

![7.png](./sample/7.png)

```
> $ python3 main.py sample/7.png
Recognized Value: 9
Confidence: [97.95279745]%
```


![1218.png](./sample/1218.png)

```
> $ python3 main.py sample/1218.png
Recognized Value: 0
Confidence: [99.99797855]%
```






