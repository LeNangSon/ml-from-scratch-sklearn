import numpy as np

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

    for _ in range(num_epochs):
         grad_w, grad_b = loss.gradient(X, w, b ,y, alpha, beta)
         
         w -= learning_rate * grad_w
         b -= learning_rate * grad_b
    return w, b

def Mini_Batch_GD(X, y, num_epochs, learning_rate, batch_size, loss, alpha=.0, beta = .0):
    num_samples = X.shape[0] 
    num_batches = (num_samples - 1) // batch_size + 1
    
    w = np.zeros(X.shape[1])
    b = 0

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
    return w, b
