
# coding: utf-8

# In[448]:


from zipfile import ZipFile
import numpy as np

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    def create_batches(self, img, lab, s = 10):
        r = np.random.randint(img.shape[0], size = s)
        return img[r], lab[r]
        pass


# In[449]:


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


# In[450]:


import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# In[466]:


# performs mini-batch SGD
# hl: hidden layer size
# alpha: learning rate
# beta: regularization parameter
# bsize: batch size for SGD
# epoch: number of iterations for training
def minisgd (hlsize = 50, alpha = 10**-5, beta = 10, bsize = 10, epoch = 5000):
    
    np.random.seed(0)
    
    print 'Loading dataset\n...'
    dl = DataLoader()
    img, lab = dl.load_data()
    u = NN(hlsize, 784, 10)
    print 'Dataset Loaded'
    
    crl_history = [0]
    regl_history = [0]
    totl_history = [0]
    epo_no = []
    
    d = 10
    l = 0
    print '\nStarting training'
    for i in range(epoch):
        ti, tl = dl.create_batches(img, lab, bsize)
        u.forward(ti)
        u.backward(tl, beta, alpha)
        crl_history[l] += u.crl
        regl_history[l] += u.regl
        totl_history[l] += u.totl
        if ((i+1) % (epoch/10) == 0):
            print '%d%% complete' %d
            d += 10
        if ((i+1) % (epoch/50) == 0):
            crl_history.append(0)
            regl_history.append(0)
            totl_history.append(0)
            epo_no.append(i+1)
            l += 1
    crl_history = [x/(epoch/50) for x in crl_history]
    regl_history = [x/(epoch/50) for x in regl_history]
    totl_history = [x/(epoch/50) for x in totl_history]
    u.forward(img)
    u.loss(lab, beta)
    crl_history[-1] = u.crl
    regl_history[-1] = u.regl
    totl_history[-1] = u.totl
    epo_no.append(i+1)
    print 'Training completed'
    
    print '\nInitially'
    print 'The cross entropy loss was: %f' % crl_history[0]
    print 'The regularization loss was: %f' % regl_history[0]
    print 'The total loss was: %f' % totl_history[0]
    print '\nAfter %d passes' % (i+1)
    print 'The cross entropy loss is: %f' % crl_history[-1]
    print 'The regularization loss is: %f' % regl_history[-1]
    print 'The total loss is: %f' % totl_history[-1]
    
    plt.plot(epo_no, crl_history, 'b-', label = 'Cross-entropy loss')
    plt.plot(epo_no, regl_history, 'g-', label = 'Regularization loss')
    plt.plot(epo_no, totl_history, 'r-', label = 'Total loss')
    plt.legend()
    plt.show()
    
    imgte, labte = dl.load_data(mode = 'test')
    u.forward(imgte)
    print '\nOn the test set:'
    u.loss(labte, beta, True)
    
    print '\nOn the train set:'
    u.forward(img)
    u.classacc(lab)
    
    print '\nOn the test set:'
    u.forward(imgte)
    u.classacc(labte)
    
    pass


# In[469]:


minisgd(hlsize = 100, alpha = 3 * 10**-6, beta = 1, bsize = 20, epoch = 6000)

