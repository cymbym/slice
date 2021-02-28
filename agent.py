from data_generate import read_csv, pos_to_state, VELOCITY, TIME, GAP_TIME, GAP_LEN, NUM_UNIT
import tensorflow as tf
import numpy as np
import time
from queue import Queue
import heapq
from log_config import Log


log = Log()
this = time.time()
LIST_PRB, LIST_PRB_COPY = [18, 29, 24, 19, 15, 8, 28, 8, 11, 17], [18, 29, 24, 19, 15, 8, 28, 8, 11, 17]
n_u, n_steps, n_input, = NUM_UNIT, GAP_LEN, 2
n_hidden = 128
n_hiddens = [n_hidden, n_hidden, n_hidden]
learning_rate = 0.01
lambda_m, lambda_k, lambda_gamma = 0.5, 0.5, 0.1
train = True
model_path ="/model/model.ckpt"
sess = tf.Session()

state_input = tf.placeholder(tf.float32, [None, n_steps, n_u, n_input], name="state_placeholder")
q_target = tf.placeholder(tf.float32, [None], name="q_target_placeholder")
reward_input = tf.placeholder(tf.float32, [None], name="reward_placeholder")
with tf.variable_scope("wb", reuse=tf.AUTO_REUSE):
    weights = {
        'in': tf.get_variable(name="w_in", dtype=tf.float32,
                              initializer=tf.truncated_normal([n_steps * n_u * n_input, n_steps * n_hidden], mean=0, stddev=0.1)),
        'out': tf.get_variable(name="w_out", dtype=tf.float32,
                               initializer=tf.truncated_normal([n_hidden, n_u], mean=0, stddev=0.1))
    }
    biases = {
        'in': tf.get_variable(name="b_in", dtype=tf.float32, initializer=tf.constant(0.1, shape=[n_steps * n_hidden])),
        'out': tf.get_variable(name="b_out", dtype=tf.float32, initializer=tf.constant(0.1, shape=[n_u]))
    }


def build_cell(num_units, keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


def dynamic_RNN(x, weights, biases):
    # x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [-1, n_steps * n_u * n_input])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden])

    with tf.variable_scope("MultiLSTM", reuse=tf.AUTO_REUSE):
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(n_h, 0) for n_h in n_hiddens])
        outputs, states = tf.nn.dynamic_rnn(multi_rnn_cell, x_in, dtype=tf.float32)
        # outputs = tf.transpose(outputs, [1, 0, 2])  # (32, ?, 128) 32个时序，取最后一个时序outputs[-1]=(?,128)
        # print(outputs.shape)
        # pred = tf.matmul(tf.gather(outputs, int(outputs.get_shape()[0]) - 1), weights['out']) + biases['out']
        # u = tf.sigmoid(pred)

    outputs = tf.reshape(outputs, [-1, n_steps, n_hidden, 1])
    return outputs


def cnn(x):
    # 定义卷积层, 16个卷积核, 卷积核大小为5，用Relu激活

    print(x.shape)
    conv0 = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu)
    print(conv0.shape)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
    print(pool0.shape)
    # 定义卷积层, 32个卷积核, 卷积核大小为5，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 32, 5, activation=tf.nn.relu)
    print(conv1.shape)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
    print(pool1.shape)
    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool1)
    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
    dropout_fc = tf.layers.dropout(fc, rate=0.5)
    dense = tf.layers.dense(dropout_fc, n_u)
    q = tf.nn.softmax(dense, name="softmax_tensor")
    return q


u = dynamic_RNN(state_input, weights, biases)
q_eval = cnn(u)
loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))
op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def assign(tmp_state, queue_index, queue_num, tmp_pos, tmp_ai, tmp_ri, i, j, k, values_action, update):
    while queue_index.qsize() >= np.sum(tmp_ai):
        list_index, list_num = queue_index.get(), queue_num.get()
        for tmp_index in range(len(list_index)):
            LIST_PRB[list_index[tmp_index]] += list_num[tmp_index]
    list_index, list_num = [], []
    list_action = heapq.nlargest(len(values_action), range(len(values_action)), values_action.__getitem__)
    index_action = 0
    reward = 0
    tmp_num = tmp_ri[k]
    if update:
        log.critical("决策前: %s" % LIST_PRB)
    while index_action < len(values_action):
        if LIST_PRB[list_action[index_action]] >= tmp_num:
            list_index.append(list_action[index_action])
            list_num.append(tmp_num)
            reward += \
                (tmp_state[0][list_action[index_action]][0] + lambda_m * tmp_num / tmp_ri[k]) * (k + 1) * lambda_k
            LIST_PRB[list_action[index_action]] -= tmp_num
            break
        elif LIST_PRB[list_action[index_action]] == 0:
            index_action += 1
        else:
            list_index.append(list_action[index_action])
            list_num.append(LIST_PRB[list_action[index_action]])
            reward += \
                (tmp_state[0][list_action[index_action]][0] + lambda_m * LIST_PRB[list_action[index_action]] / tmp_ri[k]) * (
                            k + 1) * lambda_k
            tmp_num -= LIST_PRB[list_action[index_action]]
            LIST_PRB[list_action[index_action]] = 0
            index_action += 1
    if update:
        queue_index.put(list_index)
        queue_num.put(list_num)
        log.critical("决策后: %s" % LIST_PRB)
    else:
        for tmp_index in range(len(list_index)):
            LIST_PRB[list_index[tmp_index]] += list_num[tmp_index]
    return queue_index, queue_num, reward

def plot_cost(list_cost):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(list_cost)), list_cost)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    # plt.show()
    plt.savefig('cost.jpg', transparent=True, pad_inches=0)

def learn(queue_index, queue_num, tmp_pos, tmp_ai, tmp_ri, i, j, k, list_cost):
    tmp_state = pos_to_state([tmp_pos], LIST_PRB, len_move=(j + 1) * VELOCITY)  # 100辆车 * 32个时刻 * 10个基站 * 2种指标
    log.info("获取q_target")
    values_action = sess.run(q_eval,
                             feed_dict={state_input: tmp_state})
    values_action = values_action[0]
    log.info("下一个状态的参考q值: %s" % values_action)
    queue_index, queue_num, reward = \
        assign(tmp_state[0], queue_index, queue_num, tmp_pos, tmp_ai, tmp_ri, i, j, k, values_action, update=False)
    tmp_state = pos_to_state([tmp_pos], LIST_PRB, len_move=j * VELOCITY)  # 100辆车 * 32个时刻 * 10个基站 * 2种指标
    log.info("更新参数ing")
    _, values_action, tmp_cost = sess.run([op, q_eval, loss],
                                          feed_dict={state_input: tmp_state,
                                                     q_target: [reward + lambda_gamma * np.max(values_action)]})
    values_action = values_action[0]
    log.info("当前状态的q值: %s" % values_action)
    list_cost.append(tmp_cost)
    queue_index, queue_num, reward = \
        assign(tmp_state[0], queue_index, queue_num, tmp_pos, tmp_ai, tmp_ri, i, j, k, values_action, update=True)
    return queue_index, queue_num, list_cost


def train():
    matrix_pos, matrix_ai, matrix_ri = read_csv()
    sess.run(init)
    log.debug("开始训练")
    queue_index, queue_num = Queue(), Queue()
    list_cost = []
    for idx_car in range(len(matrix_pos)):
        while not queue_index.empty():
            list_index, list_num = queue_index.get(), queue_num.get()
            for tmp_index in range(len(list_index)):
                LIST_PRB[list_index[tmp_index]] += list_num[tmp_index]
        for idx_prb in range(int(TIME / GAP_TIME)):
            for k in range(3, -1, -1):
                for l in range(matrix_ai[idx_car][k]):
                    log.debug("第%s辆车在第%s米的第%s类的第%s个。原数据：%s, %s, %s"
                              % (idx_car, idx_prb*VELOCITY, k, l+1, matrix_pos[idx_car], matrix_ai[idx_car], matrix_ri[idx_car]))
                    queue_index, queue_num, list_cost = \
                        learn(queue_index, queue_num, matrix_pos[idx_car], matrix_ai[idx_car], matrix_ri[idx_car],
                              idx_car, idx_prb, k, list_cost)
        plot_cost(list_cost)
    print(time.time() - this)
    saver.save(sess, model_path)
    print("训练结束，保存模型到{}".format(model_path))

for variable_name in tf.global_variables():
    print(variable_name)
train()
