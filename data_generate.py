import random
import math
import time
import numpy
import pandas as pd
from ast import literal_eval

VELOCITY = 25  # 25m/s=90km/h
TIME = 200
DIS_UNIT, DIS_GAP, DIS_ROAD = 500, 35, 26
NUM_UNIT = int(VELOCITY * TIME / DIS_UNIT)
PT, FC = 46, 5.8*10**9
PT, FC = 46, 5.8
GAP_LEN, GAP_TIME = 32, 1  # 窗口长32，时间间隔1s

pos_col = random.uniform(0, 26)  # 距道路右侧距离
pos_row = 0  # 距马路起始点距离。开30s，就是750m

this = time.time()
def generate_dis(x, y):
    matrix_dis = []
    for i in range(int(VELOCITY * TIME / DIS_UNIT)):
        if i * DIS_UNIT > x:
            matrix_dis.append(math.sqrt((DIS_ROAD - y + DIS_GAP) * (DIS_ROAD - y + DIS_GAP)
                                        + (i * DIS_UNIT - x) * (i * DIS_UNIT - x)))
        else:
            matrix_dis.append(math.sqrt((DIS_ROAD - y + DIS_GAP) * (DIS_ROAD - y + DIS_GAP)
                                        + (x - i * DIS_UNIT) * (x - i * DIS_UNIT)))
    return matrix_dis

def generate_sinr(dis):
    matrix_sinr = []
    for i in range(len(dis)):
        # lbf = 40.0 * math.log(dis[i], 10) + 10.5 - 18.5 * math.log(32, 10) - 18.5 * math.log(1.5, 10) \
        #       + 1.5 * math.log(FC / 5, 10)
        lbf = 21.5 * math.log(dis[i], 10) + 44.2 + 20 * math.log(FC / 5, 10)
        matrix_sinr.append(PT - lbf)
    return matrix_sinr


def generate_prb():
    for i in range(10):
        print(random.randint(0, 30), end="")
        print(", ", end="")


def generate_csv():
    matrix_pos, matrix_ai, matrix_ri = [], [], []
    for i in range(100):
        matrix_pos.append(random.randint(0, 26))  # 距道路右侧距离

        num_slice = random.randint(0, 10)
        a3 = random.randint(0, num_slice)
        num_slice -= a3
        a2 = random.randint(0, num_slice)
        num_slice -= a2
        a1 = random.randint(0, num_slice)
        a0 = num_slice - a1
        matrix_ai.append([a0, a1, a2, a3])        # 同切片类型数量

        r3 = random.randint(1, 16)
        r2 = random.randint(16, 32)
        r1 = random.randint(1, 8)
        r0 = random.randint(8, 32)
        matrix_ri.append([r0, r1, r2, r3])        # 不同切片类型需要PRB数量

    dataframe = pd.DataFrame({'matrix_pos': matrix_pos, 'matrix_ai': matrix_ai, 'matrix_ri': matrix_ri})
    print(dataframe)
    dataframe.to_csv(r"data.csv", sep=',', index=False)

def str_to_list(list_str):
    tmp_list = []
    for tmp in list_str:
        tmp_list.append(literal_eval(tmp))
    return tmp_list

def read_csv():
    f = open('data.csv', encoding='UTF-8')
    data = pd.read_csv(f)
    print(data)
    matrix_pos = data['matrix_pos'].values
    matrix_ai = str_to_list(data['matrix_ai'].values)
    matrix_ri = str_to_list(data['matrix_ri'].values)
    return matrix_pos, matrix_ai, matrix_ri


def pos_to_state(matrix_pos, list_prb, len_move):
    matrix_state = []
    for i in range(len(matrix_pos)):
        matrix_tmp = []
        for j in range(GAP_LEN):
            matrix_dis = generate_dis(j * GAP_TIME + len_move, matrix_pos[i])
            matrix_sinr = generate_sinr(matrix_dis)
            matrix_tmp.append(numpy.transpose([matrix_sinr, list_prb]).tolist())
        matrix_state.append(matrix_tmp)
    return matrix_state


# matrix_dis = generate_dis(pos_row, pos_col)
# matrix_sinr = generate_sinr(matrix_dis)
# generate_csv()
# matrix_pos, matrix_ai, matrix_ri = read_csv()
# print(time.time() - this)