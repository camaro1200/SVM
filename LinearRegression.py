import numpy as np
from numpy.random import rand

def xavier(in_, out_):
    return np.random.randn(in_, out_) * np.sqrt(2. / (in_ + out_))


class LinearRegression:
    def __init__(self, X, y, random_init=False):
        self.X = X
        self.y = y
        self.M = X.shape[0]
        self.N = X.shape[1]
        if random_init == False:
            self.w = xavier(self.N, 1)
        else:
            self.w = 
  
        
    
    def fit(self, epochs=1000, lr=0.05):
        cost_val = []
        
        for i in range(epochs):
            j = cost_function(self.X, self.y, self.w)  
            cost_val.append(j)
            #print(j)
    
            y_hat = sigmoid(np.dot(self.X, self.w))                                   
            self.w -= lr * np.dot(self.X.T,  y_hat - self.y) / self.M                 
    
        return cost_val
    
    def predict(self, X1, y1):
        pred = np.around(sigmoid(X1 @ self.w))
        acc = accuracy_score(y1, pred)
        return acc