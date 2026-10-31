import numpy as np

class BaseOptimizer:
    def __init__(self, learning_rate_init = .01):
        self.learning_rate_init = self.learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, params, grads):
        updates = self._get_updates(grads)

        for param, update in zip(params, updates):
            param += update

    def _get_updates(self,grads):
        raise NotImplementedError

class Momentum(BaseOptimizer):
    def __init__(self, params, learning_rate = .01, momentum = 0.9):
        super().__init__(learning_rate)
        self.params = params
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in params]
    def _get_updates(self, grads):
        self.velocities = [self.momentum * veclocity + (1 - self.momentum) * grad for veclocity, grad in zip(self.velocities, grads)]
        return [-self.learning_rate * velocity for velocity in self.velocities]
    """
        params = [w, b]
        grad_pred = loss.grads(y_hat, y)
        grads = ...               
        solver = Momentum(params)
        solver.update_params(params, grads)
    """

class RMSProp(BaseOptimizer):
    def __init__(self, params, learning_rate = .01, beta = 0.9, epsilon = 1e-8):
            super().__init__(learning_rate)
            self.params = params
            self.beta = beta
            self.epsilon = 1e-8
            self.squared_avg = [np.zeros_like(p) for p in params]
    def _get_updates(self, grads):
        self.squared_avg = [self.beta * veclocity + (1 - self.beta) * grad ** 2 for veclocity, grad in zip(self.squared_avg, grads)]
        return [-self.learning_rate * grad / (np.sqrt(s) + self.epsilon) for grad, s in zip(grads, self.squared_avg)]
        """
            params = [w, b]
            grad_pred = loss.grads(y_hat, y)
            grads = ...               
            solver = RMSProp(params)
            solver.update_params(params, grads)
        """
# Version 0 (Temporary)
class Adam(BaseOptimizer):
    def __init__(self, params , learning_rate = .01, momentum = .9, beta = .9, epsilon = 1e-8):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.beta = beta
        self.velocities = self.squared_avg = [np.zeros_like(p) for p in params]
        self.epsilon = epsilon
    def _get_updates(self, grads):
        self.velocities = [self.momentum * veclocity + (1 - self.momentum) * grad for veclocity, grad in zip(self.velocities, grads)]
        self.squared_avg = [self.beta * veclocity + (1 - self.beta) * grad ** 2 for veclocity, grad in zip(self.squared_avg, grads)]
        return [ -self.learning_rate * velocity /  (s + self.epsilon) for velocity, s in zip(self.velocities, self.squared_avg)]

    
# ........

def SGD(X, y, num_epochs, learning_rate, loss, alpha = .0, beta = .0):
    num_samples = X.shape[0]

    w = np.zeros(X.shape[1])
    b = 0

    losses = []

    for _ in range(num_epochs):
        shuffle = np.random.permutation(num_samples)
        X_shuffled = X[shuffle]
        y_shuffled = y[shuffle]
        for i in range(num_samples):
            X_i = X_shuffled[i: i+1]
            y_i = y_shuffled[i: i+1]

            grad_w, grad_b = loss.gradient(X_i, w, b ,y_i)
            
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

            losses.append(loss.value(X, w, b, y))

    return w, b, losses

def Batch_GD(X, y, num_epochs, learning_rate, loss, alpha = .0, beta = .0):
    w = np.zeros(X.shape[1])
    b = 0

    losses = []
    for _ in range(num_epochs):
         grad_w, grad_b = loss.gradient(X, w, b ,y)
         
         w -= learning_rate * grad_w
         b -= learning_rate * grad_b
         losses.append(loss.value(X,w,b,y))
    return w, b, losses

def Mini_Batch_GD(X, y, num_epochs, learning_rate, batch_size, loss, alpha=.0, beta = .0):
    num_samples = X.shape[0] 
    num_batches = (num_samples - 1) // batch_size + 1
    
    w = np.zeros(X.shape[1])
    b = 0
    losses = []
    for _ in range(num_epochs):
        shuffle = np.random.permutation(num_samples)
        X_shuffled = X[shuffle]
        y_shuffled = y[shuffle]

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size if (start + batch_size) < num_samples else num_samples
            X_i = X_shuffled[start: end]
            y_i = y_shuffled[start: end]

            grad_w, grad_b = loss.gradient(X_i, w, b, y_i, alpha, beta)

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

        losses.append(loss.value(X, w, b, y))

    return w, b, losses
