'''
Neural net model using stochastic gradient descent, modified for binary output (GoT death prediction)
Credit to Michael A. Nielsen, "Neural Networks and Deep Learning"
'''

import numpy as np
import random

# network class
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # for each hidden layer and the output layer, creates a list of biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # for each gap between layers (n - 1 total), generates an array of lists
        # array[0][0] is the list of weights between the first layer and the first neuron of the second layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # computes output for an input through each layer
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (activations[-1] - y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # does one step of gradient descent given a mini batch and learning rate
    def updateminibatch(self, minibatch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for each input, expected output pair in minibatch
        for x, y in minibatch:
            # computes gradient of cost function
            delta_b, delta_w = self.backprop(x, y)

            # sums all the gradients into nablas
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

        # updates biases and weights accordingly
        self.biases = [b - (eta / len(minibatch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / len(minibatch)) * nw for w, nw in zip(self.weights, nabla_w)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # trains the network via mini-batch stochastic gradient descent
    def train(self, training, epochs, minibatchsize, eta, test):
        if test:
            testn = len(test)
            
        n = len(training)
        for i in xrange(epochs):
            # shuffles training data
            random.shuffle(training)

            # makes a list of mini batches
            minibatches = [training[j:j + minibatchsize] for j in xrange(0, n, minibatchsize)]
            for minibatch in minibatches:
                self.updateminibatch(minibatch, eta)
            if test:
                # show the number of successes
                print "Epoch %d: %d / %d" % (i, self.evaluate(test), testn)
            else:
                print "Epoch %d complete" % j


    def display(self):
        print self.biases
        print "------"
        print self.weights

# computes output of network object using sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))