import sys, os, time
import numpy as np

from simple_lstm import LSTM

def gen_index(input_file):
    char2index = {}
    index2char = {}
    index = 0
    lines = []
    with open(input_file) as fp:
        for line in fp.readlines():
            if len(line.strip()) == 0:
                continue
            lines.append(line)
            for ch in line:
                if ch in char2index:
                    continue
                char2index[ch] = index
                index2char[index] = ch
                index += 1
    max_index = index
    return char2index, index2char, max_index, lines

def gen_index_seq(line, char2index, max_index):
    x_data = np.zeros((len(line), max_index+1))
    y_data = np.zeros((len(line), max_index+1))
    for i in xrange(len(line) - 1):
        ch = line[i]
        x_index = char2index.get(ch, max_index)
        x_data[i][x_index] = 1
        ch_next = line[i + 1]
        y_index = char2index.get(ch_next, max_index)
        y_data[i][y_index] = 1
        i += 1
    return x_data, y_data

def gen_char_seq(indexes, index2char):
    char_list = []
    for index in indexes:
        char_list.append(index2char.get(index, 'NULL'))
    return ''.join(char_list)


def train_model():
    print 'loading data ...'
    char2index, index2char, max_index, lines = gen_index('./data/input.txt')
    train_x_datas = []
    train_y_datas = []
    for line in lines:
        x_seq, y_seq = gen_index_seq(line, char2index, max_index)
        train_x_datas.append(x_seq)
        train_y_datas.append(y_seq)

    x_test_line = 'yo'
    x_test_seq, _ = gen_index_seq(x_test_line, char2index, max_index)
    stop_indexes = [char2index.get('\n')]
    
    #print 'x_data example:', train_x_datas[0]
    #print 'y_data example:', train_y_datas[0]

    x_dim = len(train_x_datas[0][0])
    hidden_num = 100
    out_dim = x_dim
    lstm = LSTM(x_dim, hidden_num, out_dim, 0.1, 0.00000001)

    def test_gen(lstm, x_test_seq, init_state, stop_indexes):
        seq_index = lstm.greedy_forward(x_test_seq, init_state, stop_indexes, 100)
        line_gened = gen_char_seq(seq_index, index2char)
        print 'gen:', x_test_line + line_gened

    iter = 1000
    print 'start to train ...'
    now = int(time.time())

    h_init = np.zeros((1, hidden_num))
    c_init = np.zeros((1, hidden_num))
    k = 0
    init_state = (h_init, c_init)
    for i in xrange(iter):
        for j in xrange(len(train_x_datas)):
            x_seq = train_x_datas[j]
            y_seq = train_y_datas[j]
            loss, state = lstm.train_once(x_seq, y_seq, init_state)
            if k % 10 == 0:
                print 'k', k, 'loss', loss
                print 'cost_time:', int(time.time()) - now
                now = int(time.time())
            if k % 100 == 0:
                print "iter", i, "loss:", loss
                test_gen(lstm, x_test_seq, init_state, stop_indexes) 

            k += 1
            if loss < 0.0001:
                break
    init_state = (np.zeros((1, hidden_num)), np.zeros((1, hidden_num)))
    test_gen(lstm, x_test_seq, init_state, stop_indexes) 
    print 'finished'



if __name__ == "__main__":
    train_model()
