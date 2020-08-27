import numpy as np
import sys

class NN_layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.Whidden = np.random.randn(output_size, input_size) / 1000
        self.bias_hidden = np.zeros((output_size, 1))
        self.afterForward = False
        self.last_weight_velocity = 0
        self.last_bias_velocity = 0
    
    def forward(self, X):
        self.last_input = X.copy()
        self.afterForward = True
        return self.Whidden @ X + self.bias_hidden

    def get_diff(self, dL_dy):
        return self.Whidden.T @ dL_dy
    
    def backprop(self, dL_dy, learning_rate, alpha):
        if not self.afterForward:
            print(UserWarning("This network haven't been forwarded yet"))
            sys.exit()
        dL_dWh = dL_dy @ self.last_input.T
        dL_dbh = dL_dy.copy()
        temp = self.Whidden.copy()
        self.last_weight_velocity = alpha * self.last_weight_velocity + learning_rate * dL_dWh
        self.last_bias_velocity = alpha * self.last_bias_velocity + learning_rate * dL_dbh
        self.Whidden -= self.last_weight_velocity
        self.bias_hidden -= self.last_bias_velocity
        return temp.T @ dL_dy
    
class Artificial_Neural_Network():
    def __init__(self, input_size, output_size, hiddens, activation_function, diff_activation_function):
        neuron_size = [input_size] + hiddens + [output_size]
        self.Layers = [NN_layer(neuron_size[i], neuron_size[i+1]) for i in range(len(neuron_size) - 1)]
        self.activation_function = activation_function
        self.diff_activation_function = diff_activation_function
    
    def forward(self, X):
        inp = X.copy()
        self.last_input = {0: inp.copy()}
        for layer_idx in range(len(self.Layers)):
            inp = self.activation_function(self.Layers[layer_idx].forward(inp))
            self.last_input[layer_idx+1] = inp.copy()
        out = inp.copy()
        return out
    
    def backprop(self, dL_dy, learning_rate=0.02, alpha=0.98, returns='dL_dx'):
        temp = dL_dy.copy()
        mse = temp ** 2
        for layer_idx in reversed(range(len(self.Layers))):
            dL_dy *= self.diff_activation_function(self.last_input[layer_idx + 1])
            dL_dy = self.Layers[layer_idx].backprop(dL_dy, learning_rate, alpha)
        if returns == 'dL_dx':
            return dL_dy # dL_dx
        if returns == 'mse':
            return np.sum(mse)
    
    def get_diff(self, dL_dy):
        for layer_idx in reversed(range(len(self.Layers))):
            dL_dy *= self.diff_activation_function(self.last_input[layer_idx + 1])
            dL_dy = self.Layers[layer_idx].get_diff(dL_dy)
        return dL_dy
    
    def train(self, Xtrains, Ytrains, learning_rate=0.02, alpha=0.98):
        loss = 0; _ = 0
        for x, y in zip(Xtrains, Ytrains):
            out = self.forward(x)
            dL_dy = out - y
            loss += dL_dy ** 2
            _ += 1
            self.backprop(dL_dy, learning_rate, alpha)
        return float(np.sum(loss) / _)
    
    def predict(self, X):
        return self.forward(X)

if __name__ == "__main__":
    Xtrain = np.array([[[1], [1]], [[1], [0]], [[0], [1]], [[0], [0]]])
    Ytrain = np.array([[[1]], [[0]], [[0]], [[1]]])

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    dsigmoid_dx = lambda y: y * (1 - y)

    ann = Artificial_Neural_Network(2, 1, [2], sigmoid, dsigmoid_dx)
    for epoch in range(50000):
        loss = ann.train(Xtrain, Ytrain, learning_rate=0.1, alpha=0.9)
        print("Epoch: {} || Loss: {}".format(epoch + 1, loss))