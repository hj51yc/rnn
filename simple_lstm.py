
import sys, os
import numpy as np

def sigmoid(a):
    return 1.0/(1+np.exp(-a))

def dev_sigmoid(a):
    return a*(1-a)

def tanh(a):
    return np.tanh(a)

def dtanh(a):
    return 1 - (a ** 2)

def softmax(y):
    p = np.exp(y)
    s = np.sum(p)
    final = p / s
    return final

class LSTM(object):
"""
    simple LSTM: N blocks with one cell in every block!
"""
    def __init__(self, x_dim, hiden_num,  output_dim):
        ## concate input as [h, x]
        Z = x_dim + hiden_num
        H = hiden_num
        self.Wi = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bi = np.zeros((1, H))
        self.Wf = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bf = np.zeros((1, H))
        self.Wo = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bo = np.zeros((1, H))
        self.Wc = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bc = np.zeros((1, H))

        self.Wy = np.random.randn((H, D)) / np.sqrt(D / 2.0)
        self.by = np.zeros((1, D))

    
    def forward(self, x, state):
        c_prev, h_prev = state
        X = np.column_stack((h_prev, x))
        
        hi = sigmoid(np.dot(X, self.Wi) + self.bi)
        hf = sigmoid(np.dot(X, self.Wf) + self.bf)
        ho = sigmoid(np.dot(X, self.wo) + self.bo)
        
        hc = tanh(np.dot(X, self.Wc) + self.bc)
        c = hf * c_prev + hi * hc
        h = ho * tanh(c)

        y = np.dot(h, self.Wy) + self.by
        prop = softmax(y)
        cache = (hi, hf, ho, hc, c, h, y, c_prev, h_prev, X)
        return prop, cache


    def backward(self, prob, y_label, d_next, cache)
        hi, hf, ho, hc, c, h, y, c_prev, h_prev, X = cache
        dc_next, dh_next = dnext

        ## softmax loss gradient
        dy = prob.copy()
        dy[1, y_label] -= 1

        dWy = np.dot(h.T, dy)
        dby = dy

        dh = np.dot(dy, self.Wy.T) + dh_next
        
        dho = tanh(c) * dh
        dho = dsigmoid(ho) * dho

        dc = ho * dh + dc_next
        dc = dtanh(c) * dc

        dhc = hi * dc
        dhc = dhc * dtanh(hc)

        dhf = c_prev * dc
        dhf = dsigmoid(hf) * dhf

        dhi = hc * dc
        dhi = dsigmoid(hi) * dhi

        dWf = np.dot(X.T , dhf)
        dbf = dhf
        dXf = np.dot(dhf, self.Wf.T)

        dWi = np.dot(X.T, dhi)
        dbi = dhi
        dXi = np.dot(dhi, self.Wi.T)

        dWo = np.dot(X.T, dho)
        dbo = dho
        dXo = np.dot(dho, self.Wo.T)

        dWc = np.dot(X.T, dhc)
        dbc = dhc
        dXc = np.dot(dbc, self.Wc.T)

        dX = dXf + dXi + dXo + dXc
        new_dh_next = dX[:, :H]
        
        # Gradient for c_old in c = hf * c_old + hi * hc
        new_dc_next = hf * dc

        grad = dict(Wf=dWf, Wi=dWi, Wo=dWo, Wc=dWc, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
        new_d_next = (new_dc_next, new_dh_next) 

        return new_d_next, grad

    
    def cross_entropy(self, prob, y_true):
        log_prob_neg = np.log(1 - prob)
        log_prob = np.log(prob)

        y_true_neg = 1 - y_true

        return np.sum(log_prob_neg * y_true_neg) + np.sum(log_prob * y_true)

    def train_step(self, x_seq, y_seq, state):
        probs = []
        caches = []
        loss = 0.0
        h, c = state

        for x, y in zip(x_seq, y_seq):
            prob, cache = self.forward(x, state)
            probs.append(prob)
            caches.append(cache)
            loss += cross_entropy(prob, y_true)
            
        loss /= len(x_seq.shape[0])
        
        d_next = (np.zeros_like(h), np.zeros_like(c))
        grads = {}

        for prob, y_true, cache in reverse(list(zips(probs, y_seq, caches))):
            d_next, cur_grads = self.backward(prob, y_true, cache)
            for key in cur_grads:
                if key not in grads:
                    grads[key] = cur_grads[key]
                else:
                    grads[key] += cur_grads[key]

        return grads, loss, state

    
    def adagrad(self, grads, eta, epsilon):
        for w_name in grads:
            W = getattr(self, w_name)
            dW = grads[w_name]
            if w_name in self.adagrads_sum:
                self.adagrads_sum[w_name] += dW * dW
            else:
                self.adagrads_sum[w_name] = dW * dW + epsilon

            W_step = eta / np.sqrt(self.adagrads[w_name])
            W -= W_step * dW

            
