
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from zipfile import ZipFile
import math
from sklearn.linear_model import LogisticRegression
import argparse

import matplotlib.pyplot as plt


# In[2]:


# Returns images and labels corresponding for training and testing. Default mode is train. 
# For retrieving test data pass mode as 'test' in function call.
def load_data(mode = 'train'):
    np.random.seed(12345)
    label_filename = mode + '_labels'
    image_filename = mode + '_images'
    label_zip = '../data/' + label_filename + '.zip'
    image_zip = '../data/' + image_filename + '.zip'
    with ZipFile(label_zip, 'r') as lblzip:
        labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
    with ZipFile(image_zip, 'r') as imgzip:
        images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

# Creates random batches of size s
# From labeled data img with labels lab
def create_batches(img, lab, s = 10):
    r = np.random.randint(img.shape[0], size = s)
    return img[r], lab[r]
    pass


# In[3]:


class NN:
    def __init__(self, inp = 784, hlsize = 100, outp = 10, alpha = 0.001, beta = 10):
        
        self.graph = tf.Graph()
        self.input_dim = inp
        self.output_dim = outp
        self.learning_rate = alpha
        self.hidden_dim = hlsize
        
        tf.set_random_seed(12345)

        with self.graph.as_default():
            
            # input layer
            with tf.name_scope('Input'):
                self.X = tf.placeholder(tf.float32, shape = (None, inp), name = 'images')

            # labels
            with tf.name_scope('Cross-Entropy-Loss'):
                self.Y = tf.placeholder(tf.int64, name = 'labels')
                self.oh = tf.one_hot(self.Y, outp, name = 'one-hot')

            # 3 hidden layers
            with tf.name_scope('Layer1'):
                self.W1 = tf.Variable(tf.truncated_normal((inp, hlsize), stddev = 1/math.sqrt(inp), name = 'weights'))
                self.b1 = tf.Variable(tf.zeros(hlsize), name = 'bias')
                o1 = tf.matmul(self.X, self.W1, name = 'prod')
                o1b = tf.add(o1, self.b1, name = 'biased')
                o1r = tf.maximum(o1b, 0, name = 'rectified')
            
            with tf.name_scope('Layer2'):
                self.W2 = tf.Variable(tf.truncated_normal((hlsize, hlsize), stddev = 1/math.sqrt(hlsize), name = 'weights'))
                self.b2 = tf.Variable(tf.zeros(hlsize), name = 'bias')
                o2 = tf.matmul(o1r, self.W2, name = 'prod')
                o2b = tf.add(o2, self.b2, name = 'biased')
                o2r = tf.maximum(o2b, 0, name = 'rectified')

            with tf.name_scope('Layer3'):
                self.W3 = tf.Variable(tf.truncated_normal((hlsize, hlsize), stddev = 1/math.sqrt(hlsize), name = 'weights'))
                self.b3 = tf.Variable(tf.zeros(hlsize), name = 'bias')
                o3 = tf.matmul(o2r, self.W3, name = 'prod')
                o3b = tf.add(o3, self.b3, name = 'biased')
                o3r = tf.maximum(o3b, 0, name = 'rectified')
               
            # output layer
            with tf.name_scope('Output'):
                self.W4 = tf.Variable(tf.truncated_normal((hlsize, outp), stddev = 1/math.sqrt(hlsize), name = 'weights'))
                self.b4 = tf.Variable(tf.zeros(outp), name = 'bias')
                o4 = tf.matmul(o3r, self.W4, name = 'prod')
                o4b = tf.add(o4, self.b4, name = 'biased')

            # softmax layer
            with tf.name_scope('Softmax'):
                expst = tf.exp(o4b - tf.reduce_max(o4b, axis = 1, keepdims = True), name = 'stable-exp')
                preds = tf.div(expst, tf.reduce_sum(expst, axis = 1, keepdims = True))
                self.pred = tf.add(1e-15, preds, name = 'predictions') 
            
            # loss
            with tf.name_scope('Loss'):
                self.crl = tf.reduce_mean(- tf.reduce_sum(self.oh * tf.log(self.pred), axis = 1), name = 'cross-entropy')
                self.regl = tf.multiply(0.5 * beta, tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)) + tf.reduce_sum(tf.square(self.W3)) + tf.reduce_sum(tf.square(self.W4)), name = 'regularization')
                self.cost = tf.add(self.crl, self.regl, name = 'total')
    
            # global steps
            self.global_step = tf.Variable(0, name='global_step', trainable = False)
            
            # classification accuracy
            with tf.name_scope('Classification-Accuracy'):       
                self.clpred = tf.equal(tf.argmax(self.pred, 1), self.Y)
                self.accuracy = tf.reduce_mean(tf.cast(self.clpred, tf.float32), name="Accuracy")
            
            # summary
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('cr_entropy_loss', self.crl)
            tf.summary.scalar('regln_loss', self.regl)
            tf.summary.scalar('total_loss', self.cost)
            self.merged = tf.summary.merge_all()

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, self.global_step)


# In[17]:


def train(hidden_dim = 200, learning_rate = 1e-4, regularization = 1e-2, training_epochs = 50, tr_val_split = 0.7, batch_size = 100):
    
    train_data, train_labels = load_data()
    print 'Data Loaded...\n'

    # divide data into training and cross validation sets
    val_split_len = int(tr_val_split * len(train_labels))
    val_data = train_data[val_split_len:]
    val_labels = train_labels[val_split_len:]
    train_data = train_data[:val_split_len]
    train_labels = train_labels[:val_split_len]
    print 'Training and Cross Validation Set created...\n'
    
    # create the graph, set parameters to control training
    nn = NN(train_data.shape[1], hidden_dim, np.max(train_labels) + 1, learning_rate, regularization)
    print 'Neural Network created...\n'
    
    best_validation_accuracy = 0.0 # Best validation accuracy yet seen
    last_improvement = 0 # Last epoch where validation accuracy improved
    patience = 10 # Stop optimization if no improvement found in this many iterations
    
    # Start session
    sv = tf.train.Supervisor(graph = nn.graph, logdir = 'logs/', summary_op = None, save_model_secs = 0)

    print 'Training started...\n'
    
    with sv.managed_session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        
        for epoch in range(training_epochs):
            
            print '\nTraining Epoch : %d/%d' %(epoch + 1, training_epochs)
            
            avg_cost = 0
            d = 10
            total_iters = int(len(train_data) / batch_size)
            tenth = total_iters/10
            
            if sv.should_stop(): 
                break
                
            crl = [0]
            regl = [0]
            totl = [0]
            tim = [0]
            u = 0
            for i in range(total_iters):
                batch_xs, batch_ys = create_batches (train_data, train_labels, batch_size)
                feed = {nn.X: batch_xs, nn.Y: batch_ys}
                c, _ = sess.run([nn.cost, nn.optimizer], feed)
                avg_cost += c
                
                if ((i + 1) % 50 == 0):
                    sv.summary_computed(sess, sess.run(nn.merged, feed))
                    f, g, h = sess.run([nn.crl, nn.regl, nn.cost], feed)
                    crl.append((crl[-1] * u + f)/(u + 1))
                    regl.append((regl[-1] * u + g)/(u + 1))
                    totl.append((totl[-1] * u + h)/(u + 1))
                    tim.append(i+1)
                    u += 1
                    
                if ((i + 1) % tenth == 0):
                    print '%d%% completed' %d
                    d += 10

            # computing losses
            avg_cost /= total_iters
            print '\nAverage Training Loss: %.3f' %avg_cost
            acc = sess.run(nn.accuracy, {nn.X: val_data, nn.Y: val_labels})
            print 'Validation Accuracy: %.2f%%'  %(acc*100)
            
            # graphing progress
            plt.figure()
            plt.title('Epoch %d Loss' %(epoch+1))
            plt.plot(tim[1:], crl[1:], 'b', label = 'Cross Entropy')
            plt.plot(tim[1:], regl[1:], 'r', label = 'Regularization')
            plt.plot(tim[1:], totl[1:], 'g', label = 'Total')
            plt.legend()
            plt.gca().set_ylim(bottom = 0)
            plt.savefig('graphs/Epoch %d Loss.png' %(epoch+1))

            print 'Progress Graph saved in graphs directory\n'
            
            # ensuring continuous improvement
            if acc > best_validation_accuracy:
                last_improvement = epoch
                best_validation_accuracy = acc
                gs = sess.run(nn.global_step)
                sv.saver.save(sess, 'logs/model_gs', global_step=gs)
                
            if epoch - last_improvement > patience:
                print"Early stopping ..."
                break
                
        print '\nBest Validation Accuracy achieved by network: %.2f%%\n' %(best_validation_accuracy*100)


# In[18]:


#train()


# In[19]:


def test(hdim = 200):
    
    test_data, test_labels = load_data('test')
    
    nn = NN(hlsize = hdim)
    
    with nn.graph.as_default():    
        sv = tf.train.Supervisor()        
        with sv.managed_session() as sess:
            
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint('logs/'))
            print "\nModel Restored"        

            acc = sess.run(nn.accuracy, {nn.X: test_data, nn.Y: test_labels})
            print 'Test Accuracy: %.2f%%' %(acc * 100)


# In[20]:


#test()


# In[23]:


def layer(num, hdim = 200):
    
    nn = NN(hlsize = hdim)
    
    with nn.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint('logs/'))
            print("\nModel Restored!")
            
            [w1, b1, w2, b2, w3, b3] = sess.run([nn.W1, nn.b1, nn.W2, nn.b2, nn.W3, nn.b3])

            train_data, train_labels = load_data()
            test_data, test_labels = load_data("test")
            
            if num >= 1:
                neural_train = np.add(np.matmul(train_data, w1),b1)
                neural_test = np.add(np.matmul(test_data, w1),b1)
                
            if num >= 2:
                neural_train = np.add(np.matmul(neural_train, w2),b2)
                neural_test = np.add(np.matmul(neural_test, w2),b2)

            if num == 3:
                neural_train = np.add(np.matmul(neural_train, w3),b3)
                neural_test = np.add(np.matmul(neural_test, w3),b3)

            LR = LogisticRegression (max_iter = 20, solver = 'sag', warm_start = True, n_jobs = -1, verbose = 5)
            print "\nTraining Logistic Regression on Layer %d...\n" %num
            LR.fit (neural_train, train_labels)
            
            print "\nTesting ..."
            score = LR.score(neural_test, test_labels)
            print "Logistic Regression accuracy : %.2f%%" %(score * 100)


# In[24]:


#layer(1)


# In[25]:


#layer(2)


# In[26]:


#layer(3)


# In[27]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--layer', type = int)
    
    args = parser.parse_args()
    
    hidden_dim = 200
    regularization = 1e-2
    training_epochs = 10
    learning_rate = 1e-4
    tr_val_split = 0.7
    
    if args.train:
        train(hidden_dim, learning_rate, regularization, training_epochs, tr_val_split)
    elif args.test:
        test(hidden_dim)
    elif args.layer != None:
        layer(args.layer)
    else:
        print 'Default Mode: Training\n'
        train()


# In[29]:


if __name__ == "__main__":
    main()


