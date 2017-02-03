import numpy as np
import sys, os


def sigmoid(a):
    return 1.0/(1+np.exp(-a))

def dev_sigmoid(a):
    return a*(1-a)

class RNN(object):
    
    def __init__(self, input_dim, out_dim, hiden_dim, alpha=0.01):
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hiden_dim = hiden_dim
        self.Whh = np.random.random((hiden_dim, hiden_dim)) * 2 - 1
        self.Why = np.random.random((hiden_dim, out_dim)) * 2 - 1
        self.Wxh = np.random.random((input_dim, hiden_dim)) * 2 - 1
        self.bias_h = np.random.random((1, hiden_dim))
        self.bias_y = np.random.random((1, out_dim))
        self.alpha = alpha

    
    def train(self, inputs, outputs):
        for i in xrange(len(inputs)):
            input = inputs[i]
            output = outputs[i]
            error = self.forward(input, output)
            if i % 100 == 0:
                print 'error', error
            self.backward(input, output)
    
    
    def train_once(self, input, output):
        error, pred = self.forward(input, output)
        self.backforward(input, output)
        return error, pred


    def forward(self, input, output):
        self.h_list = []
        self.h_list.append(np.zeros((1, self.hiden_dim)))
        self.net_h_list = []
        self.theta_y_list = []
        error = 0.0
        pred = []
        for t in xrange(len(input)):
            x = input[t]
            h_prev = self.h_list[t]
            net_h = np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bias_h
            self.net_h_list.append(net_h)
            h = sigmoid(net_h)
            self.h_list.append(h)
            net_y = np.dot(h, self.Why) + self.bias_y
            p = sigmoid(net_y)
            y = output[t]
            theta_y_t = (p - y) * dev_sigmoid(p)
            self.theta_y_list.append(theta_y_t)
            error += abs(p - y)
            pred.append(p)
        return error, pred


    def backforward(self, input, output):
        h_theta_next = np.zeros((1, self.hiden_dim))
        d_Whh = np.zeros((self.hiden_dim, self.hiden_dim))
        d_Wxh = np.zeros((self.input_dim, self.hiden_dim))
        d_Why = np.zeros((self.hiden_dim, self.out_dim))
        d_bias_h = np.zeros((1, self.hiden_dim))
        d_bias_y = np.zeros((1, self.out_dim))
        input_len = len(input)
        for p in xrange(input_len):
            t = input_len - p - 1
            #print  't:', t
            x = input[t]
            y = output[t]
            h_cur = self.h_list[t + 1]
            h_prev = self.h_list[t]
            theta_y_t = self.theta_y_list[t]
            part_1 = (np.dot(self.Why, theta_y_t.T) + np.dot(self.Whh, h_theta_next.T)).T 
            theta_h_t = part_1 * dev_sigmoid(h_cur) 
            h_theta_next = theta_h_t
            #print 'part_1.shape', part_1.shape
            #print 'theta_h_t.shape', theta_h_t.shape
            #print 'theta_y_t.shape', theta_y_t.shape
            #print 'Why.shape', self.Why.shape
            #print 'Whh.shape', self.Whh.shape
            #print 'h_theta_next.shape', h_theta_next.shape
            #print 'net_h_cur.shape', net_h_cur.shape
            d_Why += np.dot(h_cur.T, theta_y_t)
            #print 'theta_y_t', theta_y_t
            #print 'y.T', y.T
            #print 'd_Wxh', d_Wxh
            d_Whh += np.dot(h_prev.T, theta_h_t)
            d_Wxh += np.dot(x.T, theta_h_t)
            d_bias_h += theta_h_t
            d_bias_y += theta_y_t
        
        self.Whh -= d_Whh * self.alpha
        self.Wxh -= d_Wxh * self.alpha
        self.Why -= d_Why * self.alpha
        self.bias_h -= d_bias_h * self.alpha
        self.bias_y -= d_bias_y * self.alpha
        #print 'd_Why', d_Why
        #print 'Why', self.Why
        #print 'Wxh', self.Wxh
        #print 'Why', self.Why
        #print 'bias_h', self.bias_h
        #print 'bias_y', self.bias_y


if __name__ == '__main__':
    rnn = RNN(2, 1, 16, 0.1)
    binary_dim = 8
    largest_number = pow(2,binary_dim)
    binary_codes = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    
    for i in xrange(20000):
        a_int = np.random.randint(largest_number/2)
        b_int = np.random.randint(largest_number/2)
        c_int = a_int + b_int
        a = binary_codes[a_int]
        b = binary_codes[b_int]
        c = binary_codes[c_int]
        input = []
        output = []
        for k in xrange(binary_dim):
            p = binary_dim - k - 1
            input.append(np.array([[a[p],b[p]]]))
            output.append(np.array([[c[p],]]))

        error, pred = rnn.train_once(input, output)
        #if i == 5:
        #    sys.exit(0)
        if i % 1000 == 0:
            print 'i', i
            print 'error', error
            print 'true', output
            print 'pred', pred
    
    print 'finished'

