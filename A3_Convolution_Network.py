#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

class Convolution:
    def __init__(self, num_filters = 3, stride = 1, pad = 1, alpha=0.01):
        self.num_filters = num_filters
        self.stride     = stride
        self.filter     = np.random.randn(self.num_filters, self.num_filters)
        self.filter     = self.filter/self.filter.sum()
        self.bias       = np.random.rand()/10
        self.pad        = pad
        self.alpha      = alpha

    def convolving(self, X, fil, dimen_x, dimen_y):
        z = np.zeros((dimen_x, dimen_y))
        for i in range(dimen_x):
            for ii in range(dimen_y):
                temp = np.multiply(X[i : i+fil.shape[0], ii : ii+fil.shape[1]], fil)
                z[i,ii] = temp.sum()
        return z
        
    def forward_pass(self, X):
        self.X    = X
        (d, p, t) = self.X.shape
        dimen_x   = int(((p - self.num_filters)/self.stride) + 1)
        dimen_y   = int(((t - self.num_filters)/self.stride) + 1)
        self.z    = np.zeros((d, dimen_x, dimen_y))
        for i in range(d):
            self.z[i] = (self.convolving(self.X[i], self.filter, dimen_x, dimen_y) + self.bias)

        return self.z

    def backward(self, grad_z):
        (d, p, t)  = grad_z.shape
        filter_1   = np.flip((np.flip(self.filter, axis = 0)), axis = 1)
        self.grads = np.zeros((d, p, t))
        for i in range(d):
            self.grads[i] = self.convolving(np.pad(grad_z[i], ((1,1), (1,1)), 'constant', constant_values = 0), filter_1, p, t)

        self.grads = np.pad(self.grads, ((0,0),(1,1),(1,1)), 'constant', constant_values = 0)

        self.grad_filter = np.zeros((self.num_filters, self.num_filters))

        for i in range(self.num_filters):
            for ii in range(self.num_filters):
                self.grad_filter[i, ii] = (np.multiply(grad_z, self.X[:, i:p+i, ii:t+ii])).sum()
        self.grad_filter = self.grad_filter/(d)

        self.grad_bias = (grad_z.sum())/(d)
        return self.grads

    def applying_sgd(self):
        self.filter = self.filter - (self.alpha*self.grad_filter)
        self.bias   = self.bias - (self.alpha*self.grad_bias)

class MaxPool2:
    def __init__(self, pool_dim = 2, stride = 2):
        self.pool_dim = pool_dim
        self.stride = stride

    def forward_pass(self, data):
        (q, p, t) = data.shape
        z_x = int((p - self.pool_dim) / self.stride) + 1
        z_y = int((t - self.pool_dim) / self.stride) + 1
        after_pool = np.zeros((q, z_x, z_y))
        for ii in range(0, q):
            liss = []
            for i in range(0,p,self.stride):
                for j in range(0,t,self.stride):
                    if (i+self.pool_dim <= p) and (j+self.pool_dim <= t):
                        temp = data[ii, i:(i+(self.pool_dim)), j:(j+(self.pool_dim))]
                        temp_1 = np.max(temp)
                        liss.append(temp_1)
            liss = np.asarray(liss)
            liss = liss.reshape((z_x, z_y))
            after_pool[ii] = liss
            del liss
        return after_pool

    def backward(self, pooled):
        (a,b,c) = pooled.shape   
        cheated = np.zeros((a,2*b,2*c))
        for k in range(0, a):
            pooled_transpose_re = pooled[k].reshape((b*c))
            count = 0
            for i in range(0, 2*b, self.stride):
                for j in range(0, 2*c, self.stride):
                    cheated[k, i:(i+(self.stride)),j:(j+(self.stride))] = pooled_transpose_re[count]
                    count = count+1
        return cheated

    def applying_sgd(self):
        pass

fac = 5
class Fully_Connected:
    def __init__(self, in_dim, out_dim, alpha = 0.01, Theta = None, bias = None):
        self.alpha = alpha
        if Theta == None:
            self.weights = np.random.randn(in_dim, out_dim)/fac

        else:
            self.weights = Theta

        if bias == None:
            self.bias = np.random.randn(out_dim)/fac

        else:
            self.bias = bias

    def forward_pass(self, X):
        self.X = X
        self.z = np.matmul(X, self.weights) + self.bias
        return self.z

    def backward(self, grad_previous):
        t= self.X.shape[0]
        self.grad = np.matmul((self.X.transpose()), grad_previous)/t
        self.grad_bias = (grad_previous.sum(axis=0))/t
        self.grad_a = np.matmul(grad_previous, self.weights.transpose())
        return self.grad_a

    def applying_sgd(self):
        self.weights = self.weights - (self.alpha*self.grad)
        self.bias = self.bias - (self.alpha*self.grad_bias)

class softmax:
    def __init__(self):
        pass
    
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    
    def forward_pass(self, z):
        self.z =  z
        (p,t) = self.z.shape
        self.a = np.zeros((p,t))
        for i in range(0,p):
            for ii in range(0,t):
                self.a[i,ii] = (np.exp(self.z[i,ii]))/(np.sum(np.exp(self.z[i,:])))
        return self.a

    def backward(self, Y):
        y = self.expansion(Y)
        self.grad = (self.a - y)
        return self.grad

    def applying_sgd(self):
        pass

class relu:
    def __init__(self):
        pass

    def forward_pass(self, z):
        
        if (len(z.shape) == 3):

            z_temp = z.reshape((z.shape[0], z.shape[1]*z.shape[2]))
            z_temp_1 = self.forward_pass(z_temp)
            self.a_1 = z_temp_1.reshape((z.shape[0], z.shape[1], z.shape[2]))
            return (self.a_1)

        else:
            (p,t) = z.shape
            self.a = np.zeros((p,t))
            for i in range(0,p):
                for ii in range(0,t):
                        self.a[i,ii] = max([0,z[i,ii]])
            return self.a

    def derivative(self, a):
        if a>0:
            return 1
        else:
            return 0
    
    def backward(self, grad_previous):
        
        if (len(grad_previous.shape)==3):

            (d, p, t) = grad_previous.shape
            self.grad = np.zeros((d, p, t))
            
            for i in range(d):
                for ii in range(p):
                    for iii in range(t):
                        self.grad[i, ii, iii] = (grad_previous[i, ii, iii] * self.derivative(self.a_1[i, ii, iii]))
            
            return (self.grad)

        else:
            (p,t) = grad_previous.shape
            self.grad = np.zeros((p,t))
            for i in range(p):
                for ii in range(t):
                    self.grad[i,ii] = grad_previous[i,ii] * self.derivative(self.a[i,ii])
            return (self.grad)

    def applying_sgd(self):
        pass

class padding():    
    def __init__(self, pad = 1):
        self.pad = pad

    def forward_pass(self, data):
        X = np.pad(data , ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),'constant', constant_values=0)
        return X

    def backward(self, y):
        return (y[:, 1:(y.shape[1]-1),1:(y.shape[2]-1)])

    def applying_sgd(self):
        pass

class CNN:
    def __init__(self, Network):
        self.Network = Network

    def forward_pass(self, X):
        n = X
        for i in self.Network:
            n = i.forward_pass(n)
            
        return n
    
    def backward(self, Y):
        m = Y
        count = 1
        for i in (reversed(self.Network)):
            m = i.backward(m)

    def update_params(self):
        for i in self.Network:
            i.applying_sgd()

class reshaping:    
    def __init__(self):
        pass

    def forward_pass(self, a):
        self.shape_a = a.shape
        
        self.final_a = a.reshape(self.shape_a[0], self.shape_a[1]*self.shape_a[2])
        return self.final_a
    
    def backward(self, q):
        return (q.reshape(self.shape_a[0], self.shape_a[1], self.shape_a[2]))

    def applying_sgd(self):
        pass

class cross_entropy:
    def __init__(self):
        pass
    
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y

    def loss(self, A, Y):
        exp_Y = self.expansion(Y)
        (u,i) = A.shape
        loss_matrix = np.zeros((u,i))
        for j in range(u):
            for jj in range(i):
                if exp_Y[j,jj] == 0:
                    loss_matrix[j,jj] = np.log(1 - A[j,jj])
                else:
                    loss_matrix[j,jj] = np.log(A[j,jj])
        
        return ((-(loss_matrix.sum()))/u)

class accuracy:
    def __init__(self):
        pass

    def value(self, out, Y):
        self.out = np.argmax(out, axis=1)
        p = self.out.shape[0]
        total = 0
        for i in range(p):
            if Y[i]==self.out[i]:
                total += 1
        return total/p

import tensorflow.keras.datasets.fashion_mnist as fashion_mnist

# Load data from tensorflow
data = fashion_mnist.load_data()

(X_train, y_train), (X_test, y_test) = data
plt.imshow(X_train[0])
plt.show()

# Reduce the sample size
from sklearn.model_selection import train_test_split
train_size = 50000
test_size  = 10000
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size,
                                                  test_size=test_size, shuffle=True)

print("Raw Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

# Preprocess data
# Normalize data within {0,1}
X_train = X_train / 255.0
X_val   = X_val / 255.0
X_test  = X_test / 255.0

# actural item corresponding to each label
item = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# visualizing the first 10 images in the dataset and their labels
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.title(item[y_train[i]])
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.show()
print('label for each of the above image: %s' % (y_train[0:20]))

epochs      = 3
lr          = 0.3
stopper     = 85.0
broke       = 0
batch_size  = 5000
output_size = n_classes = len(np.unique(y_train))

# Print data shape
print("Train/Test/Validation Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

# Build Model
model = CNN([padding(),
             Convolution(),
             MaxPool2(),
             relu(),
             padding(),
             Convolution(),
             MaxPool2(),
             relu(),
             reshaping(),
             Fully_Connected(7*7, 24, alpha = lr),
             relu(),
             Fully_Connected(24, 10, alpha = lr),
             softmax()
             ])

CE = cross_entropy()
acc = accuracy()

# Train the CNN for 3 epochs
for epoch in range(epochs):
    k = 0
    for batch_idx in range(batch_size, 50000, batch_size):
        print("batch_idx :", batch_idx)
        out = model.forward_pass(X_train[k:batch_idx])
        print("epoch:{} \t batch: {} \t loss: \t {}".format(epoch+1, int(batch_idx/batch_size), CE.loss(out, y_train[k:batch_idx])), end="\t")
        accuracy = acc.value(out, y_train[k:batch_idx])*100
        print("accuracy: {}".format(accuracy))
        
        if accuracy >= stopper:
            broke = 1
            break
        model.backward(y_train[k:batch_idx])
        model.update_params()
        k = batch_idx
        
    if broke == 1:
        break
    

out = model.forward_pass(X_train)
print("The final loss is {}".format(CE.loss(out, y_train)))
print("The final accuracy on train set is {}".format(acc.value(out, y_train)*100))

X_valst = X_test/255
#X_valst = X_val.reshape((10000,28*28))/255
out_1 = model.forward_pass(X_valst)
print("The accuracy on test set is {}".format(acc.value(out_1, y_test)*100))
