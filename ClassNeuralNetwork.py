import numpy as np


class NeuralNet :
    def __init__(self, X, W, V):
        self.X = X
        self.W = W
        self.V = V
        
    def FeedForward(self):
        def af(x):
            Y = np.tanh(x)
            return Y
        net_h = self.W.T @ self.X
        Y_head = af(net_h)
        output = self.V.T @ Y_head
        # .T mean tranpose matrix
        return output

    def cost_function(self):
        #X1 = np.array([0.3, 0.35, 0.4, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0 ])
        #X2 = np.array([0.3, 0.4, 0.5, 0.75, 0.7, 0.8, 0.4, 0.5, 0.5 ])
        Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

        #self.X = np.vstack((X1, X2))
        Y_head = self.FeedForward()
        e = Y_head - Y
        #print(Y, Y_head, e)
        J = 1/9*np.sum(e**2)
        return J

"""
# Run 
W = np.random.rand(2,5) * 2 - 1 # --------> 2x5 > 5x2 * 2x9 = 5x9
V = np.random.rand(5,1) * 2 - 1 # --------> 5x1 > 1x5 * 5x9 = 1x9
X1 = np.array([0.3, 0.35, 0.4, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0 ])
X2 = np.array([0.3, 0.4, 0.5, 0.75, 0.7, 0.8, 0.4, 0.5, 0.5 ])

X = np.vstack((X1,X2)) # ----> Input matrix
print(X.shape)
Neural = NeuralNet(X,W,V)
FeedForward = Neural.FeedForward()
print(FeedForward)
J = Neural.cost_function()
print(J)"
"""
