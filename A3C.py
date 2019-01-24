# coding:utf-8


import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os
import random
import re
import pandas as pd
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils import plot_model
from keras import losses
from collections import deque
import tensorflow as tf
from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import animation
import sys
sys.setrecursionlimit(6000)
time = time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない

Nmai_mahjong = 14

shanten = Shanten()
pailist =[]
tehailist = []
senpai = []
result = 0
syanten = 0
paisID = np.zeros(136)

# -- constants of Game
NUM_STATES = 137
NUM_ACTIONS = NUM_STATES - 1
NONE_STATE = np.zeros(NUM_STATES)

# -- constants of LocalBrain
MIN_BATCH = 5
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
LEARNING_RATE = 7e-4
RMSPropDecaly = 0.99

# -- params of Advantage-ベルマン方程式
GAMMA = 0.6
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

N_WORKERS = 50   # スレッドの数
Tmax = 10   # 各スレッドの更新ステップ間隔

# ε-greedyのパラメータ
EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = 200*N_WORKERS

class Mahjong():
    ripaitehai = []

    def pais(self, pailist):
        for num in range(0, 136):
            pailist.append(num)
        paisID = np.full(136, 0)

        return paisID, pailist

    def yamatumi(self, pailist):
        returnyama = deque()
        senpai = random.sample(pailist, len(pailist))
        for i in senpai:
            returnyama.append(i)
        paisID = np.full(136, 0)
        return returnyama, paisID, pailist

    def haipai(self, yama, paisID):
        tehai = []
        for i in range(0, Nmai_mahjong - 1):
            a = yama.popleft()
            tehai.append(a)
            paisID[a] = 1
        tehai.sort()
        return tehai, yama, paisID

    def tumo(self, tehai, yama, paisID):
        a = yama.popleft()
        #print('ツモ：' + str(self.henkan(a)))
        #resultfile.write('ツモ：' + str(mahjong.henkan(a)) + '\n')
        tehai.append(a)
        paisID[a] = 1

        tehai.sort()

        return tehai, yama, paisID

    def dahai(self, tehai, action, kawa, paisID):
        a = tehai.pop(action)
        #print(action)
        #print('打：' + str(self.henkan(a)))
        #resultfile.write('打：' + str(mahjong.henkan(a)) + '\n\n')
        kawa.append(a)
        paisID[a] = 0

        tehai.sort()

        return tehai, kawa, paisID

    def richi(self, tehai, kawa, paisID):
        a = tehai.pop(Nmai_mahjong - 1)
        #print('ツモ切り：' + str(self.henkan(a)))
        #resultfile.write('ツモ切り：' + str(mahjong.henkan(a)) + '\n\n')
        kawa.append(a)
        paisID[a] = 0

        tehai.sort()

        return tehai, kawa, paisID

    def randomdahai(self, tehai, kawa, paisID):
        x = 0
        count = 0
        b = random.randint(0, Nmai_mahjong - 1)
        while b == count:
            if paisID[x] == 1:
                count += 1
            x += 1

        #print(b)
        a = tehai.pop(b)
        #print('ランダム打：' + str(self.henkan(a)))
        #resultfile.write('ランダム打：' + str(mahjong.henkan(a)) + '\n\n')
        kawa.append(a)
        paisID[a] = 0
        tehai.sort()

        return tehai, kawa, paisID, a

    def syanten(self, tehais):
        _, m, p, s, h = self.remove_mpsh(self.henkan(tehais))
        tiles = TilesConverter.string_to_34_array(man=m, pin=p, sou=s, honors=h)
        result, a, b = shanten.calculate_shanten(tiles)

        return min(result, a, b)

    def remove_mpsh(self, tile):
        cpytehai = []
        man = []
        pin = []
        sou = []
        honors = []
        man2 = []
        pin2 = []
        sou2 = []
        honors2 = []

        cpytehai.extend(tile)
        for n in range(len(cpytehai)):
            pai = cpytehai.pop()
            if 'm' in str(pai):
                man.append(pai)
            elif 'p' in str(pai):
                pin.append(pai)
            elif 's' in str(pai):
                sou.append(pai)
            else:
                honors.append(pai)

        for a in range(len(man)):
            man2.append(re.sub('m', '', man.pop()))
        for a in range(len(pin)):
            pin2.append(re.sub('p', '', pin.pop()))
        for a in range(len(sou)):
            sou2.append(re.sub('s', '', sou.pop()))
        for a in range(len(honors)):
            honors2.append(re.sub('z', '', honors.pop()))

        result = sorted(man2) + sorted(pin2) + sorted(sou2) + sorted(honors2)

        return result, man2, pin2, sou2, honors2

    def henkan(self, pais):
        henkanpais = []

        if type(pais) == list:
            for i in pais:
                if i == 0 or i == 1 or i == 2 or i == 3:
                    henkanpais.append("1m")
                if i == 4 or i == 5 or i == 6 or i == 7:
                    henkanpais.append("2m")
                if i == 8 or i == 9 or i == 10 or i == 11:
                    henkanpais.append("3m")
                if i == 12 or i == 13 or i == 14 or i == 15:
                    henkanpais.append("4m")
                if i == 16 or i == 17 or i == 18 or i == 19:
                    henkanpais.append("5m")
                if i == 20 or i == 21 or i == 22 or i == 23:
                    henkanpais.append("6m")
                if i == 24 or i == 25 or i == 26 or i == 27:
                    henkanpais.append("7m")
                if i == 28 or i == 29 or i == 30 or i == 31:
                    henkanpais.append("8m")
                if i == 32 or i == 33 or i == 34 or i == 35:
                    henkanpais.append("9m")

                if i == 36 or i == 37 or i == 38 or i == 39:
                    henkanpais.append("1p")
                if i == 40 or i == 41 or i == 42 or i == 43:
                    henkanpais.append("2p")
                if i == 44 or i == 45 or i == 46 or i == 47:
                    henkanpais.append("3p")
                if i == 48 or i == 49 or i == 50 or i == 51:
                    henkanpais.append("4p")
                if i == 52 or i == 53 or i == 54 or i == 55:
                    henkanpais.append("5p")
                if i == 56 or i == 57 or i == 58 or i == 59:
                    henkanpais.append("6p")
                if i == 60 or i == 61 or i == 62 or i == 63:
                    henkanpais.append("7p")
                if i == 64 or i == 65 or i == 66 or i == 67:
                    henkanpais.append("8p")
                if i == 68 or i == 69 or i == 70 or i == 71:
                    henkanpais.append("9p")

                if i == 72 or i == 73 or i == 74 or i == 75:
                    henkanpais.append("1s")
                if i == 76 or i == 77 or i == 78 or i == 79:
                    henkanpais.append("2s")
                if i == 80 or i == 81 or i == 82 or i == 83:
                    henkanpais.append("3s")
                if i == 84 or i == 85 or i == 86 or i == 87:
                    henkanpais.append("4s")
                if i == 88 or i == 89 or i == 90 or i == 91:
                    henkanpais.append("5s")
                if i == 92 or i == 93 or i == 94 or i == 95:
                    henkanpais.append("6s")
                if i == 96 or i == 97 or i == 98 or i == 99:
                    henkanpais.append("7s")
                if i == 100 or i == 101 or i == 102 or i == 103:
                    henkanpais.append("8s")
                if i == 104 or i == 105 or i == 106 or i == 107:
                    henkanpais.append("9s")

                if i == 108 or i == 109 or i == 110 or i == 111:
                    henkanpais.append("1z")
                if i == 112 or i == 113 or i == 114 or i == 115:
                    henkanpais.append("2z")
                if i == 116 or i == 117 or i == 118 or i == 119:
                    henkanpais.append("3z")
                if i == 120 or i == 121 or i == 122 or i == 123:
                    henkanpais.append("4z")
                if i == 124 or i == 125 or i == 126 or i == 127:
                    henkanpais.append("5z")
                if i == 128 or i == 129 or i == 130 or i == 131:
                    henkanpais.append("6z")
                if i == 132 or i == 133 or i == 134 or i == 135:
                    henkanpais.append("7z")
        else:
           if pais == 0 or pais == 1 or pais == 2 or pais == 3:
               henkanpais.append("1m")
           if pais == 4 or pais == 5 or pais == 6 or pais == 7:
               henkanpais.append("2m")
           if pais == 8 or pais == 9 or pais == 10 or pais == 11:
               henkanpais.append("3m")
           if pais == 12 or pais == 13 or pais == 14 or pais == 15:
               henkanpais.append("4m")
           if pais == 16 or pais == 17 or pais == 18 or pais == 19:
               henkanpais.append("5m")
           if pais == 20 or pais == 21 or pais == 22 or pais == 23:
               henkanpais.append("6m")
           if pais == 24 or pais == 25 or pais == 26 or pais == 27:
               henkanpais.append("7m")
           if pais == 28 or pais == 29 or pais == 30 or pais == 31:
               henkanpais.append("8m")
           if pais == 32 or pais == 33 or pais == 34 or pais == 35:
               henkanpais.append("9m")

           if pais == 36 or pais == 37 or pais == 38 or pais == 39:
               henkanpais.append("1p")
           if pais == 40 or pais == 41 or pais == 42 or pais == 43:
               henkanpais.append("2p")
           if pais == 44 or pais == 45 or pais == 46 or pais == 47:
               henkanpais.append("3p")
           if pais == 48 or pais == 49 or pais == 50 or pais == 51:
               henkanpais.append("4p")
           if pais == 52 or pais == 53 or pais == 54 or pais == 55:
               henkanpais.append("5p")
           if pais == 56 or pais == 57 or pais == 58 or pais == 59:
               henkanpais.append("6p")
           if pais == 60 or pais == 61 or pais == 62 or pais == 63:
               henkanpais.append("7p")
           if pais == 64 or pais == 65 or pais == 66 or pais == 67:
               henkanpais.append("8p")
           if pais == 68 or pais == 69 or pais == 70 or pais == 71:
               henkanpais.append("9p")

           if pais == 72 or pais == 73 or pais == 74 or pais == 75:
               henkanpais.append("1s")
           if pais == 76 or pais == 77 or pais == 78 or pais == 79:
               henkanpais.append("2s")
           if pais == 80 or pais == 81 or pais == 82 or pais == 83:
               henkanpais.append("3s")
           if pais == 84 or pais == 85 or pais == 86 or pais == 87:
               henkanpais.append("4s")
           if pais == 88 or pais == 89 or pais == 90 or pais == 91:
               henkanpais.append("5s")
           if pais == 92 or pais == 93 or pais == 94 or pais == 95:
               henkanpais.append("6s")
           if pais == 96 or pais == 97 or pais == 98 or pais == 99:
               henkanpais.append("7s")
           if pais == 100 or pais == 101 or pais == 102 or pais == 103:
               henkanpais.append("8s")
           if pais == 104 or pais == 105 or pais == 106 or pais == 107:
               henkanpais.append("9s")

           if pais == 108 or pais == 109 or pais == 110 or pais == 111:
               henkanpais.append("1z")
           if pais == 112 or pais == 113 or pais == 114 or pais == 115:
               henkanpais.append("2z")
           if pais == 116 or pais == 117 or pais == 118 or pais == 119:
               henkanpais.append("3z")
           if pais == 120 or pais == 121 or pais == 122 or pais == 123:
               henkanpais.append("4z")
           if pais == 124 or pais == 125 or pais == 126 or pais == 127:
               henkanpais.append("5z")
           if pais == 128 or pais == 129 or pais == 130 or pais == 131:
               henkanpais.append("6z")
           if pais == 132 or pais == 133 or pais == 134 or pais == 135:
               henkanpais.append("7z")

        return henkanpais

    def make_state(self, paisID, t):
        state = np.reshape(paisID, (1, 136))

        t_arry = np.asarray([[t]])
        state = np.hstack((state, t_arry))
        return state

    def make_tehai(self, state):
        tehai = []
        for i in range(0, 136):
            if state[i] == 1:
                 tehai.append(i)

        return tehai


# --グローバルなTensorFlowのDeep Neural Networkのクラスです　-------
class ParameterServer:
    def __init__(self):
        with tf.variable_scope("parameter_server"):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self.model = self._build_model()            # ニューラルネットワークの形を決定

        # serverのパラメータを宣言
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSPropDecaly)    # loss関数を最小化していくoptimizerの定義です

    # 関数名がアンダースコア2つから始まるものは「外部から参照されない関数」、「1つは基本的に参照しない関数」という意味
    def _build_model(self):     # Kerasでネットワークの形を定義します
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(100, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        plot_model(model, to_file='A3C.png', show_shapes=True)  # Qネットワークの可視化
        return model


# --各スレッドで走るTensorFlowのDeep Neural Networkのクラスです　-------
class LocalBrain:
    def __init__(self, name, parameter_server):   # globalなparameter_serverをメンバ変数として持つ
        with tf.name_scope(name):
            self.train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
            K.set_session(SESS)
            self.model = self._build_model()  # ニューラルネットワークの形を決定
            self._build_graph(name, parameter_server)  # ネットワークの学習やメソッドを定義

    def _build_model(self):     # Kerasでネットワークの形を定義します
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(100, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, name, parameter_server):      # TensorFlowでネットワークの重みをどう学習させるのかを定義します
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))  # placeholderは変数が格納される予定地となります
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = self.model(self.s_t)

        #print(p)

        # loss関数を定義します
        log_prob = tf.log(tf.clip_by_value(tf.reduce_sum(p * self.a_t, axis=1, keep_dims=True) + 1e-10, 1e-10, 1.0))
        #log_prob = tf.log(tf.reduce_sum(p * self.a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = self.r_t - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)  # stop_gradientでadvantageは定数として扱います
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(tf.clip_by_value((p + 1e-10), 1e-10, 1.0)), axis=1, keep_dims=True)  # maximize entropy (regularization)
        #entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        # 重みの変数を定義
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)  # パラメータを宣言
        # 勾配を取得する定義
        self.grads = tf.gradients(self.loss_total, self.weights_params)

        # ParameterServerの重み変数を更新する定義(zipで各変数ごとに計算)
        self.update_global_weight_params = \
            parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

        # PrameterServerの重み変数の値を、localBrainにコピーする定義
        self.pull_global_weight_params = [l_p.assign(g_p)
                                          for l_p, g_p in zip(self.weights_params, parameter_server.weights_params)]

        # localBrainの重み変数の値を、PrameterServerにコピーする定義
        self.push_local_weight_params = [g_p.assign(l_p)
                                          for g_p, l_p in zip(parameter_server.weights_params, self.weights_params)]

    def pull_parameter_server(self):  # localスレッドがglobalの重みを取得する
        SESS.run(self.pull_global_weight_params)

    def push_parameter_server(self):  # localスレッドの重みをglobalにコピーする
        SESS.run(self.push_local_weight_params)

    def update_parameter_server(self):     # localbrainの勾配でParameterServerの重みを学習・更新します
        if len(self.train_queue[0]) < MIN_BATCH:    # データがたまっていない場合は更新しない
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]
        s = np.vstack(s)    # vstackはvertical-stackで縦方向に行列を連結、いまはただのベクトル転置操作
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        # Nステップあとの状態s_から、その先得られるであろう時間割引総報酬vを求めます
        _, v = self.model.predict(s_)

        # N-1ステップあとまでの時間割引総報酬rに、Nから先に得られるであろう総報酬vに割引N乗したものを足します
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}     # 重みの更新に使用するデータ
        SESS.run(self.update_global_weight_params, feed_dict)   # ParameterServerの重みを更新

    def predict_p(self, s):    # 状態sから各actionの確率pベクトルを返します
        p, v = self.model.predict(s)
        return p

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)


# --行動を決定するクラスです、CartPoleであれば、棒付き台車そのものになります　-------
class Agent:
    def __init__(self, name, parameter_server):
        self.brain = LocalBrain(name, parameter_server)   # 行動を決定するための脳（ニューラルネットワーク）
        self.memory = []        # s,a,r,s_の保存メモリ、　used for n_step return
        self.R = 0.             # 時間割引した、「いまからNステップ分あとまで」の総報酬R

    def act(self, s):

        p = self.brain.predict_p(s)
        # a = np.argmax(p)  # これだと確率最大の行動を、毎回選択

        a = np.random.choice(NUM_ACTIONS, p=p[0])
        # probability = p のこのコードだと、確率p[0]にしたがって、行動を選択
        # pにはいろいろな情報が入っていますが確率のベクトルは要素0番目

        return p

    def advantage_push_local_brain(self, s, a, r, s_):   # advantageを考慮したs,a,r,s_をbrainに与える
        def get_sample(memory, n):  # advantageを考慮し、メモリからnステップ後の状態とnステップ後までのRを取得する関数
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
        self.R = (self.R + r * GAMMA_N) / GAMMA     # r0はあとで引き算している、この式はヤロミルさんのサイトを参照

        # advantageを考慮しながら、LocalBrainに経験を入力する
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0  # 次の試行に向けて0にしておく

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]     # # r0を引き算
            self.memory.pop(0)

class Environment:

    def __init__(self, name, thread_type, parameter_server, ID):
        global isLearned
        self.name = name
        self.thread_type = thread_type
        self.ID = ID
        self.agent = Agent(name, parameter_server)    # 環境内で行動するagentを生成
        self.mahjong = Mahjong()

        self.Q_tilist = pd.DataFrame(columns=['kyoku', 'junme', 'Pai', 'Q_value'])
        self.add_Q_tilist = pd.DataFrame(columns=['kyoku', 'junme', 'Pai', 'Q_value'])
        self.goal_average_reward = 0.9  # この報酬を超えると学習終了
        self.num_consecutive_iterations = 100  # 学習完了評価の平均計算を行う試行回数
        self.total_reward_vec = np.zeros(self.num_consecutive_iterations)  # 各試行の報酬を格納

        self.pailist = []
        self.tehailist = []
        self.senpai = []
        self.result = 0
        self.syanten = 0
        self.paisID = np.zeros(136)

        self.paisID, self.pailist = self.mahjong.pais(self.pailist)

        self.tenpai_count = 0
        self.agari_count = 0
        self.test_tenpai_count = 0
        self.test_agari_count = 0
        self.tenpai_heikin = 0
        self.test_tenpai_heikin = 0
        self.agari_heikin = 0
        self.test_agari_heikin = 0
        self.epsilon = 1.0
        if not(isLearned):
            self.e_decay = 0.1
        else:
            self.e_decay = 0.001
        self.e_min = 0.01
        self.y = []
        self.ok_flag = 1
        self.episode_reward = 0
        self.yama = deque()
        self.yama.clear()
        self.save_yama = deque()
        self.yama, self.paisID, self.pailist = self.mahjong.yamatumi(self.pailist)
        self.save_yama = self.yama.copy()
        self.max100kai = 0.0000000

    def run(self):
        self.agent.brain.pull_parameter_server()  # ParameterSeverの重みを自身のLocalBrainにコピー
        global frames  # セッション全体での試行数、global変数を書き換える場合は、関数内でglobal宣言が必要です
        global alltenpai_count
        global alltenpai_junme
        global isLearned
        global check_list
        global resultfile

        tehai = []
        tehailist = []
        kawa = []
        i = 0
        reward = 0.0
        haitei = 0
        richi = 0
        richi_sengen = 0

        done = False

        """
        if self.thread_type is  'test':
            yama, self.paisID, self.pailist = self.mahjong.yamatumi(self.pailist)
        else:
            yama = self.save_yama.copy()
        """

        yama, self.paisID, self.pailist = self.mahjong.yamatumi(self.pailist)

        self.paisID = np.full(136, 0)
        tehai, yama, self.paisID = self.mahjong.haipai(yama, self.paisID)
        tehai, yama, self.paisID = self.mahjong.tumo(tehai, yama, self.paisID)
        backsyanten = self.mahjong.syanten(tehai)
        state = self.mahjong.make_state(self.paisID, 1)
        t = 0

        #print(str(frames + 1) + '半荘目')

        while len(yama) >= 0:
            t += 1
            action = 0
            tehai_action = -1
            max = -1.000000000000000

            #if isLearned and self.thread_type is 'test':
            #if(self.name == 'local_thread2'):
            #print(self.name + '手牌' + str(self.mahjong.henkan(tehai)))
            #print(self.name + '手牌' + str(tehai))

            #print(self.paisID)

            syanten = self.mahjong.syanten(tehai)

            probability = self.agent.act(self.mahjong.make_state(self.paisID, t))

            for x in range(0, 136):
                if self.paisID[x] == 1:
                    #print(x)
                    #print(probability[0][x])
                    if probability[0][x] > max:
                        max = probability[0][x]
                        action = x


            tehai_action = tehai.index(action)
            #print('aciton :' + str(tehai_action))

            if self.epsilon <= np.random.uniform(0, 1):
                tehai, kawa, self.paisID = self.mahjong.dahai(tehai, tehai_action, kawa, self.paisID)

            else:
                tehai, kawa, self.paisID, action = self.mahjong.randomdahai(tehai, kawa, self.paisID)

            #print(state)

            """
            if backsyanten > syanten:
                reward = 1
            """
            #reward = -1

            if syanten == 0:
                done = True
                #print('聴牌!')
                reward = 100
                self.tenpai_count += 1
                self.tenpai_heikin += (t + 1)
            if self.epsilon > self.e_min:
                self.epsilon -= self.e_decay

            next_state = self.mahjong.make_state(self.paisID, t)
            tehai, yama, self.paisID = self.mahjong.tumo(tehai, yama, self.paisID)
            backsyanten = self.mahjong.syanten(tehai)

            if self.thread_type is not 'test':
                if len(yama) == 0:
                    haitei = 1
                    done = True
            else:
                if t == 18:
                    done = True

            if done:
                next_state = None

            #print('報酬' + str(reward))

            self.episode_reward += reward  # 合計報酬を更新]

            #print('state' + str(state))
            #print('next_state' + str(next_state))


            # Advantageを考慮した報酬と経験を、localBrainにプッシュ
            self.agent.advantage_push_local_brain(deepcopy(state), action, reward, deepcopy(next_state))

            state = self.mahjong.make_state(self.paisID, t)  # 状態更新

            if done:  # 終了時がTmaxごとに、parameterServerの重みを更新し、それをコピーする
                self.agent.brain.update_parameter_server()
                self.agent.brain.pull_parameter_server()
                if (syanten == 0):
                    self.total_reward_vec = np.hstack((self.total_reward_vec[1:], 1))
                    alltenpai_count = np.append(alltenpai_count, 1)
                    alltenpai_junme = np.append(alltenpai_junme, t)
                elif(syanten != 0):
                    self.total_reward_vec = np.hstack((self.total_reward_vec[1:], 0))
                    alltenpai_count = np.append(alltenpai_count, 0)
                break

        frames += 1



        # 総試行数、スレッド名、今回の報酬を出力
        #print("スレッド："+self.name)
        #print('sum : ' + str(self.episode_reward))
        #print('流局')
        print(self.name + '直近100半荘の聴牌率' + str(self.total_reward_vec.mean()))
        if self.thread_type is 'test':
            if self.total_reward_vec.mean() > self.max100kai and self.epsilon < 0.5:
                self.max100kai = self.total_reward_vec.mean()
            #print('直近100半荘の聴牌率の最高値' + str(self.max100kai))
        resultfile.write(str(frames) + ',' + str(alltenpai_count.mean()) + ',' + str(alltenpai_junme.mean()) + '\n')

        if frames % 10000 == 0:
            print('かかった半荘数' + str(frames))
            print('全スレッドの聴牌率' + str(alltenpai_count.mean()))
            print('全スレッドの平均和了順目' + str(alltenpai_junme.mean()))
        #print('e=' + str(self.epsilon))

        if self.total_reward_vec.mean() >= self.goal_average_reward and self.thread_type is not 'test':
            check_list[self.ID] = 1
            self.agent.brain.push_parameter_server()    # この成功したスレッドのパラメータをparameter-serverに渡します

        # スレッドで平均報酬が一定を越えたら終了
        if check_list.mean() >= 0.5:
            if not (isLearned):
                #print('かかった半荘数' + str(frames))
                frames = 0
                alltenpai_count = np.zeros(1)
                isLearned = True
                time.sleep(2.0)     # この間に他のlearningスレッドが止まります


# --スレッドになるクラスです　-------
class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, thread_name, thread_type, parameter_server, ID):
        self.environment = Environment(thread_name, thread_type, parameter_server, ID)
        self.thread_type = thread_type

    def run(self):
        while True:
            if not(isLearned) and self.thread_type is 'learning':     # learning threadが走る
                self.environment.run()

            if not(isLearned) and self.thread_type is 'test':    # test threadを止めておく
                time.sleep(1.0)

            if isLearned and self.thread_type is 'learning':     # learning threadを止めておく
                time.sleep(3.0)

            if isLearned and self.thread_type is 'test':     # test threadが走る
                self.environment.run()


# -- main ここからメイン関数です------------------------------
# M0.global変数の定義と、セッションの開始です
frames = 0              # 全スレッドで共有して使用する総ステップ数
alltenpai_count = np.zeros(1)
alltenpai_junme = np.zeros(1)
check_list = np.zeros(N_WORKERS)

isLearned = False       # 学習が終了したことを示すフラグ
SESS = tf.Session()     # TensorFlowのセッション開始
filepath = str('result/result' + datetime.now().strftime("%m%d %H%M"))
resultfile = open(filepath + '.txt', 'w')

# M1.スレッドを作成します
with tf.device("/cpu:0"):
    parameter_server = ParameterServer()    # 全スレッドで共有するパラメータを持つエンティティです
    threads = []     # 並列して走るスレッド
    # 学習するスレッドを用意
    for i in range(N_WORKERS):
        thread_name = "local_thread"+str(i+1)
        threads.append(Worker_thread(thread_name=thread_name, thread_type="learning", parameter_server=parameter_server, ID=i))

    # 学習後にテストで走るスレッドを用意
    threads.append(Worker_thread(thread_name="test_thread", thread_type="test", parameter_server=parameter_server, ID=-1))

# M2.TensorFlowでマルチスレッドを実行します
COORD = tf.train.Coordinator()                  # TensorFlowでマルチスレッドにするための準備です
SESS.run(tf.global_variables_initializer())     # TensorFlowを使う場合、最初に変数初期化をして、実行します

running_threads = []
for worker in threads:
    job = lambda: worker.run()      # この辺は、マルチスレッドを走らせる作法だと思って良い
    t = threading.Thread(target=job)
    t.start()
    #running_threads.append(t)

# M3.スレッドの終了を合わせます
#COORD.join(running_threads)