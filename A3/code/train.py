import numpy as np
import data_loader
import module

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


'''Implement mini-batch SGD here'''

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


minisgd(hlsize = 100, alpha = 3 * 10**-6, beta = 1, bsize = 20, epoch = 6000)
