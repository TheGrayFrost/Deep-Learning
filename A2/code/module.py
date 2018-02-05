import numpy as np

def ReLU(x):
    x[x < 0] = 0
    return x

class NN(object):
    
    # random initiation
    def __init__(self, h, inp, op):
        self.inp = inp
        self.hn = h
        self.op = op
        self.W1 = np.random.normal(0, 0.01, (inp+1, h))
        self.W2 = np.random.normal(0, 0.01, (h+1, op))
        pass
    
    # forward pass
    def forward(self, inp):
        self.inter = []
        self.inter.append(inp)
        h = np.append(inp, np.ones((inp.shape[0], 1)), axis = 1)
        self.inter.append(h)
        
        # hidden layer
        r = ReLU(np.matmul(h, self.W1))
        self.inter.append(r)
        v = np.append(r, np.ones((r.shape[0], 1)), axis = 1)
        self.inter.append(v)
        
        # output layer
        t = np.matmul(v, self.W2)
        self.inter.append(t)
        
        # softmax
        y = np.exp(t)
        y = y / np.sum(y, axis = 1, keepdims = True)
        self.inter.append(y)
        
        self.pred = y
        pass
    
    # prints the total loss
    def printloss(self):
        print 'The cross entropy loss is: %f' % self.crl
        print 'The regularization loss is: %f' % self.regl
        print 'The total loss is: %f' % self.totl
        
    # total loss
    # beta is regularization coefficient
    # outp is to set whether to print the loss
    def loss(self, lab, beta = 1, outp = False):
        
        # cross entropy loss
        u = np.zeros((lab.shape[0], self.op))
        u[np.arange(lab.shape[0]), lab] = 1
        self.exp = u
        self.crl = - np.mean((u * np.log(self.pred)) + ((1 - u) * np.log(1 - self.pred)))
        
        # L2 regularization loss   
        self.regl = 0.5 * beta * (np.mean(np.square(self.W1[:-1,:])) + np.sum(np.square(self.W2[:-1,:])))
        
        # total loss
        self.totl = self.crl + self.regl
        
        # print if required
        if outp:
            print 'The cross entropy loss is: %f' % self.crl
            print 'The regularization loss is: %f' % self.regl
            print 'The total loss is: %f' % self.totl
            
        pass     
        
    # backward pass
    # beta is regularization coefficient
    # alpha is learning rate
    # outp is to set whether to print the loss
    def backward(self, lab, beta = 1, alpha = 0.1, outp = False):
        
        # loss caculation
        self.loss (lab, beta, outp)
        
        self.der = []
        # backprop over cross entropy loss
        self.der.append((self.pred - self.exp) / (self.pred * (1 - self.pred)))
        
        # backprop over softmax
        self.der.append (self.pred * (self.der[0] - np.sum(self.pred * self.der[0], axis = 1, keepdims = True)))
        
        # backprop over output layer
        # weights derivative
        self.der.append (np.matmul(self.inter[3].T, self.der[1]))
        # input derivative
        self.der.append (np.matmul(self.der[1], self.W2.T))
        self.der.append (self.der[3][:,:-1])
        
        # backprop over hidden layer
        # weights derivative
        self.der.append (np.matmul(self.inter[1].T, self.der[4] * (self.inter[2] > 0)))
        
        # L2 derivatives        
        self.der.append (beta * np.append(self.W2[:-1,:], np.zeros((1, self.op)), axis = 0))
        self.der.append (beta * np.append(self.W1[:-1,:], np.zeros((1, self.hn)), axis = 0))
        
        # final total derivatives
        self.W1der = self.der[5] + self.der[7]
        self.W2der = self.der[2] + self.der[6]
        
        # changing weights accordingly
        self.W1 = self.W1 - alpha * self.W1der
        self.W2 = self.W2 - alpha * self.W2der
        del self.der
        pass
    
    # prints the classification accuracy
    def classacc (self, lab):
        count = 0
        h = (np.argmax(self.pred, axis = 1) == lab)
        for x in h:
            if x:
                count += 1
        print 'The classification accuracy is: %d / %d = %.2f%%' %(count, h.shape[0], (1.0 * count / h.shape[0])*100)
