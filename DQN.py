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
from time import time

Nmai_mahjong = 5

shanten = Shanten()
pailist =[]
tehailist = []
senpai = []
result = 0
syanten = 0
paisID = np.zeros([136, 3])

class Mahjong():
    ripaitehai = []
    def pais(self):
        global paisID
        for num in range(0, 136):
            pailist.append(num)
        paisID = np.full((136, 3), (1, 0, 0))

    def yamatumi(self):
        global paisID
        returnyama = deque()
        senpai = random.sample(pailist, len(pailist))
        for i in senpai:
            returnyama.append(i)
        paisID = np.full((136, 3), (1, 0, 0))
        return returnyama

    def haipai(self, yama):
        global paisID
        for i in range(0, Nmai_mahjong - 1):
            a = yama.popleft()
            tehai.append(a)
            paisID[a][0] = 0
            paisID[a][1] = 1
            paisID[a][2] = 0
        tehai.sort()
        return tehai, yama

    def tumo(self, tehai, yama):
        global paisID
        a = yama.popleft()
        resultfile.write('ツモ：' + str(mahjong.henkan(a)) + '\n')
        tehai.append(a)
        paisID[a][0] = 0
        paisID[a][1] = 1
        paisID[a][2] = 0

        return tehai, yama

    def dahai(self, tehai, action, kawa):
        global paisID
        a = tehai.pop(action)
        resultfile.write('打：' + str(mahjong.henkan(a)) + '\n')
        kawa.append(a)
        paisID[a][0] = 0
        paisID[a][1] = 0
        paisID[a][2] = 1
        tehai.sort()

        return tehai, kawa

    def richi(self, tehai, kawa):
        global paisID
        a = tehai.pop(Nmai_mahjong - 1)
        resultfile.write('ツモ切り：' + str(mahjong.henkan(a)) + '\n')
        kawa.append(a)
        paisID[a][0] = 0
        paisID[a][1] = 0
        paisID[a][2] = 1
        tehai.sort()

        return tehai, kawa

    def randomdahai(self, tehai, kawa):
        global paisID
        a = tehai.pop(random.randint(0, Nmai_mahjong - 1))
        resultfile.write('ランダム打：' + str(mahjong.henkan(a)) + '\n')
        kawa.append(a)
        paisID[a][0] = 0
        paisID[a][1] = 0
        paisID[a][2] = 1
        tehai.sort()

        return tehai, kawa

    def syanten(self, tehais):
        _, m, p, s, h =mahjong.remove_mpsh(mahjong.henkan(tehais))
        tiles = TilesConverter.string_to_34_array(man=m, pin=p, sou=s, honors=h)
        result, _, _ = shanten.calculate_shanten(tiles)

        return result

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

    def make_state(self):
        global paisID
        state = np.reshape(paisID, (1, 408))
        return state

    def make_tehai(self, state):
        tehai = []
        for i in range(0, 408, 3):
            if state[0][i+1] == 1:
                 tehai.append(i // 3)

        return tehai

class QNetwork:
    def __init__(self,hidden_size, learning_rate, state_size=408, action_size=5):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        plot_model(self.model, to_file='DQN.png', show_shapes=True)  # Qネットワークの可視化
        self.model = multi_gpu_model(self.model, gpus=1)
        self.model.compile(loss=losses.mean_absolute_error, optimizer='RMSprop')


    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 408))
        targets = np.zeros((batch_size, 5))
        mini_batch = deepcopy(memory.sample(batch_size))

        for i, (state_b, action_b, reward_b, next_state_b, haitei, richi_sengen) in enumerate(mini_batch):
            if richi_sengen != 1:
                inputs[i:i + 1] = state_b
                target = reward_b

                if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                    # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）

                    if haitei == 1:
                        target = reward_b
                    else:
                        tehai = mahjong.make_tehai(next_state_b)

                        copystate = next_state_b

                        retTargetQs = targetQN.model.predict(copystate)[0]
                        next_action = np.argmax(retTargetQs)
                        target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

                targets[i] = self.model.predict(state_b)    # Qネットワークの出力
                targets[i][action_b] = target
                self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


# [4]カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, targetQN):

        retTargetQs = targetQN.model.predict(state)[0]
        action = np.argmax(retTargetQs)

        return action, retTargetQs


# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
DQN_MODE = 0    # 1がDQN、0がDDQNです


num_episodes = 10000 # 学習回数
test_episodes = 100000 #テスト回数
total_reward_vec = np.zeros(num_episodes * 17)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数

# ---
hidden_size = 512              # Q-networkの隠れ層のニューロンの数
learning_rate = 0.00001         # Q-networkの学習係数
memory_size = 100000000            # バッファーメモリの大きさ
batch_size = 128              # Q-networkを更新するバッチの大記載

# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
#plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory = Memory(max_size=memory_size)
actor = Actor()
mahjong = Mahjong()
mahjong.pais()
Q_tilist = pd.DataFrame(columns=['kyoku', 'junme', 'Pai', 'Q_value'])
add_Q_tilist = pd.DataFrame(columns=['kyoku', 'junme', 'Pai', 'Q_value'])

tenpai_count = 0
agari_count = 0
test_tenpai_count = 0
test_agari_count = 0
tenpai_heikin= 0
test_tenpai_heikin = 0
agari_heikin= 0
test_agari_heikin = 0
epsilon = 1.0
e_decay = 0.0005
e_min = 0.01

filepath = str('result/result' + datetime.now().strftime("%m%d %H%M"))

resultfile = open(filepath + '.txt', 'w')

# [5.3]メインルーチン--------------------------------------------------------
for episode in range(num_episodes):  # 試行数分繰り返す
    yama = deque()
    yama.clear()
    tehai = []
    tehailist = []
    kawa = []
    print('\n東' + str(episode+1) + '局')
    resultfile.write('東' + str(episode+1) + '局\n')
    i = 0
    reward = 0.0
    haitei = 0
    richi = 0
    richi_sengen = 0

    done = False
    yama = mahjong.yamatumi()
    tehai, yama = mahjong.haipai(yama)
    resultfile.write('配牌' + str(mahjong.henkan(tehai)) + '\n')
    backsyanten = mahjong.syanten(tehai)
    tehai, yama = mahjong.tumo(tehai, yama)
    state = mahjong.make_state()

    episode_reward = 0

    targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
    for t in range(18):  # 1試行のループ
        max = -100
        action = -100
        i = 0
        j = 0
        count = 0

        resultfile.write('手牌' + str(mahjong.henkan(tehai)) + '\n')
        resultfile.write(str(backsyanten) + 'シャンテン' + '\n')

        action, Qti = actor.get_action(mahjong.make_state(), mainQN)

        add_Q_tilist['kyoku'] = episode + 1
        add_Q_tilist['junme'] = t + 1
        add_Q_tilist['Pai'] = mahjong.henkan(tehai)
        add_Q_tilist['Q_value'] = Qti
        Q_tilist = Q_tilist.append(add_Q_tilist)

        '''
        if richi == 1:
            richi_sengen = 0
            tehai, kawa = mahjong.richi(tehai, kawa)
        else:
        '''

        if epsilon <= np.random.uniform(0, 1):
            tehai, kawa = mahjong.dahai(tehai, action, kawa)
        else:
            tehai, kawa = mahjong.randomdahai(tehai, kawa)

        syanten = mahjong.syanten(tehai)

        resultfile.write(str(syanten) + 'シャンテン' + '\n\n')

        if syanten < backsyanten:
            reward = 0.5
        elif syanten > backsyanten:
            reward = -0.5
        else:
            reward = 0

        if syanten == 0:
            done = True
            resultfile.write('聴牌った！！\n')
            resultfile.write(str(mahjong.henkan(tehai)))
            print('聴牌')
            print(mahjong.henkan(tehai))
            print(str(t + 1) + '順目')
            tenpai_count += 1
            tenpai_heikin += (t + 1)
            reward = 1

        tehai, yama = mahjong.tumo(tehai, yama)
        next_state = mahjong.make_state()

        """
        if mahjong.syanten(tehai) == -1:
            done = True
            reward = 100
            haitei = 1
            resultfile.write('和了った！！！\n')
            resultfile.write(str(mahjong.henkan(tehai)))
            print('和了った！！！！')
            print(mahjong.henkan(tehai))
            print(str(t + 1) + '順目')
            agari_count += 1
            agari_heikin += (t + 1)
        """

        if t == 17:
            haitei = 1

        resultfile.write('報酬：' + str(reward) + '\n')

        episode_reward += reward  # 合計報酬を更新]

        memory.add((deepcopy(state), action, reward, deepcopy(next_state), haitei, richi_sengen))     # メモリの更新する

        state = mahjong.make_state()  # 状態更新
        backsyanten = syanten

        if len(yama) <= 14:
            done = True
        # Qネットワークの重みを学習・更新する replay
        if (memory.len() > batch_size * 50):
            mainQN.replay(memory, batch_size, gamma, targetQN)
            if epsilon > e_min :
                epsilon -= e_decay

        total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
        if done:
            break

    print('sum : ' + str(total_reward_vec.sum()))
    print('流局')
    print('学習中聴牌率' + str((tenpai_count / (episode + 1)) * 100))
    print('学習中和了率' + str((agari_count / (episode + 1)) * 100))
    print('e=' + str(epsilon))

resultfile.write('聴牌回数' + str(tenpai_count))
resultfile.write('聴牌率' + str((tenpai_count / (episode + 1)) * 100))
resultfile.write('ここから本番')

for episode in range(test_episodes):  # 試行数分繰り返す
    yama = deque()
    yama.clear()
    tehai = []
    tehailist = []
    kawa = []
    print('\n東' + str(episode+1) + '局')
    resultfile.write('東' + str(episode+1) + '局\n')
    i = 0
    reward = 0
    haitei = 0
    richi = 0

    done = False
    yama = mahjong.yamatumi()
    tehai, yama = mahjong.haipai(yama)
    resultfile.write('配牌' + str(mahjong.henkan(tehai)) + '\n')
    tehai, yama = mahjong.tumo(tehai, yama)
    backsyanten = mahjong.syanten(tehai)
    state = mahjong.make_state()
    episode_reward = 0

    for t in range(18):  # 1試行のループ
        max = -100
        action = -100
        i = 0
        j = 0
        count = 0

        resultfile.write('手牌' + str(mahjong.henkan(tehai)) + '\n')
        resultfile.write(str(backsyanten) + 'シャンテン' + '\n')

        action, Qti = actor.get_action(mahjong.make_state(), mainQN)

        add_Q_tilist['kyoku'] = episode + 1
        add_Q_tilist['junme'] = t + 1
        add_Q_tilist['Pai'] = mahjong.henkan(tehai)
        add_Q_tilist['Q_value'] = Qti
        Q_tilist = Q_tilist.append(add_Q_tilist)

        if richi == 1:
            richi_sengen = 0
            tehai, kawa = mahjong.richi(tehai, kawa)
        else:
            tehai, kawa = mahjong.dahai(tehai, action, kawa)

        syanten = mahjong.syanten(tehai)

        resultfile.write(str(syanten) + 'シャンテン' + '\n\n')

        if syanten == 0:
            done = True
            resultfile.write('聴牌った！！\n')
            resultfile.write(str(mahjong.henkan(tehai)))
            print('聴牌')
            print(mahjong.henkan(tehai))
            print(str(t + 1) + '順目')
            tenpai_count += 1
            tenpai_heikin += (t + 1)
            reward = 1

        tehai, yama = mahjong.tumo(tehai, yama)
        next_state = mahjong.make_state()

        if mahjong.syanten(tehai) == -1:
            done = True
            reward = 100
            haitei = 1
            resultfile.write('和了った！！！\n')
            resultfile.write(str(mahjong.henkan(tehai)))
            print('和了った！！！！')
            print(mahjong.henkan(tehai))
            print(str(t + 1) + '順目')
            agari_count += 1
            agari_heikin += (t + 1)

        if t == 17:
            haitei = 1

        if len(yama) <= 14:
            done = True

        if done:
            break

    print('流局')
    print('聴牌率' + str((test_tenpai_count / (episode + 1)) * 100))
    print('和了率' + str((test_agari_count / (episode + 1)) * 100))

print(str(test_tenpai_count)+'回テンパった')
print(str(test_agari_count)+'回和了った')
if test_tenpai_count != 0 and test_agari_count != 0:
    print('平均聴牌順目は' + str(test_tenpai_heikin / test_tenpai_count))
    print('平均和了順目は' + str(test_agari_heikin / test_agari_count))

resultfile.write('聴牌回数' + str(test_tenpai_count))
resultfile.write('聴牌率' + str((test_tenpai_count / (episode + 1)) * 100))
resultfile.write('和了回数' + str(test_agari_count))
resultfile.write('和了率' + str((test_agari_count / (episode + 1)) * 100))
Q_tilist.to_csv(filepath + '.csv')
resultfile.close()