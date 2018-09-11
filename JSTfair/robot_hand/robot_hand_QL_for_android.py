# coding: utf-8
import numpy as np
import androidhelper as android
import cv
import time


def serialStart():
     '''ArduinoとAndroidのシリアル通信'''
     droid = android.Android()
     enable = droid.usbserialHostEnable()
     l = droid.usbserialGetDeviceList().result.items()
     tk = str(l).split(',')[-1]
     h = tk.split(chr(34))[1]
     ret = droid.usbserialConnect(str(h))
     uuid = str(ret.result.split(chr(34))[-2])
     print('uuid: ', uuid)
     time.sleep(3)
     active = droid.usbserialActiveConnections()
     print('active: ', active)
     return droid, uuid


'''定数の設定'''
NUM_DIZITIZED = 6  # 各状態の離散値への分割数
discount = 0.99  # 時間割引率
lr = 0.5  # 学習係数
MAX_STEPS = 200  # 1試行のstep数cartpoleは5steps立ち続ければ終わり
NUM_EPISODES = 20  # 最大試行回数
AREA_THRESH = 90  # 赤色物体面積の閾値．0~100で規格化してある

'''Androidの設定'''
path = '/storage/7E9B-5A00/Picture/'  # カメラで撮影した画像を保存するフォルダ
droid, uuid = serialStart()  # serial通信オープン


class Arduino:
    '''Arduino制御のモータを動かす'''
    '''
    # DCモーター(車輪)
    前進: 49 / 後退: 50 / 右旋回: 51 / 左旋回: 52 / ブレーキ: 53(else)
    # サーボ(4本指)
    0度: 60 (握っている状態) / 30度: 61 / 60度: 62 / 90度: else (手を開いた状態)
    '''
    def __init__(self):
        droid.usbserialWrite((u'1'.encode('utf-8')), uuid)  # まず停止信号を出すとうまくいく(?)
        time.sleep(3)

    def go(self):
        print('go')
        droid.usbserialWrite((u'49'.encode('utf-8')), uuid)

    def back(self):
        print('back')
        droid.usbserialWrite((u'50'.encode('utf-8')), uuid)

    def right(self):
        print('right')
        droid.usbserialWrite((u'51'.encode('utf-8')), uuid)

    def left(self):
        print('left')
        droid.usbserialWrite((u'52'.encode('utf-8')), uuid)

    def stop(self):
        print('stop')
        droid.usbserialWrite((u'53'.encode('utf-8')), uuid)

    def hold(self):
        print('hold')
        droid.usbserialWrite((u'60'.encode('utf-8')), uuid)

    def release(self):
        print('release')
        droid.usbserialWrite((u'63'.encode('utf-8')), uuid)

    def thirty_deg(self):
        print('thirty_deg')
        droid.usbserialWrite((u'61'.encode('utf-8')), uuid)

    def sixty_deg(self):
        print('sixty_deg')
        droid.usbserialWrite((u'62'.encode('utf-8')), uuid)


class Agent:
    '''CartPoleのエージェントクラスです、棒付き台車そのものになります'''

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための頭脳を生成

    def update_Q_function(self, observation, action, reward, observation_next):
        '''Q関数の更新'''
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        '''行動の決定'''
        action = self.brain.decide_action(observation, step)
        return action


class Brain:
    '''エージェントが持つ脳となるクラスです、Q学習を実行します'''

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # ロボットハンドの行動（前後左右握る）の5を取得
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZED**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        '''観測した状態（連続値）を離散値にデジタル変換する閾値を求める'''
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        '''観測したobservation状態を、離散値に変換する'''
        area_sum, area_v, center_pos_x, center_pos_y, center_v_x, center_v_y = observation
        digitized = [
            np.digitize(area_sum, bins=self.bins(0, 10.0, NUM_DIZITIZED)), #　面積の比率
            np.digitize(area_v, bins=self.bins(-10.0, 10.0, NUM_DIZITIZED)),
            np.digitize(center_pos_x, bins=self.bins(-5.0, 5.0, NUM_DIZITIZED)),
            np.digitize(center_pos_y, bins=self.bins(-5.0, 5.0, NUM_DIZITIZED)),
            np.digitize(center_v_x, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED)),
            np.digitize(center_v_y, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)]) #　6進数で表して計算を圧縮

    def update_Q_table(self, observation, action, reward, observation_next):
        '''QテーブルをQ学習により更新'''
        state = self.digitize_state(observation)  # 状態を離散化
        state_next = self.digitize_state(observation_next)  # 次の状態を離散化
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + lr * (reward + discount * Max_Q_next - self.q_table[state, action])

    def decide_action(self, observation, episode):
        '''ε-greedy法で徐々に最適行動のみを採用する'''
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)  # 0,1の行動をランダムに返す
        return action


class  Environment:
    '''CartPoleを実行する環境のクラスです'''

    def __init__(self):
        num_states = 6  # 課題の状態の数(面積と重心と、それぞれの変化量)
        num_actions = 6  # ロボットハンドの行動（前進，後退，右旋回，左旋回，握る，離す）
        self.agent = Agent(num_states, num_actions)  # 環境内で行動するAgentを生成

    def cameraCapture(self, filename):
        '''Androidで写真を撮って保存する'''
        droid.cameraCapturePicture(path + str(filename) + '.png')
        img = cv.imread(path + str(filename) + '.png')
        return img

    def red_detect(self, img):
        '''赤色のマスク'''
        # HSV色空間に変換
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # 赤色のHSVの値域1
        hsv_min = np.array([0, 127, 0])
        hsv_max = np.array([30, 255, 255])
        mask1 = cv.inRange(hsv, hsv_min, hsv_max)
        # 赤色のHSVの値域2
        hsv_min = np.array([150, 127, 0])
        hsv_max = np.array([179, 255, 255])
        mask2 = cv.inRange(hsv, hsv_min, hsv_max)
        return mask1 + mask2

    def calc_area(self, img):
        '''面積計算'''
        img = self.red_detect(img)
        pix_area = cv.countNonZero(img)  # ピクセル数
        # パーセントを算出
        h, w = img.shape  # frameの面積
        per = round(100 * float(pix_area) / (w * h), 3)  # 0-100で規格化
        return pix_area, per

    def calc_center(self, img):
        '''重心座標(x,y)を求める'''
        img = self.red_detect(img)
        mu = cv.moments(img, False)
        x, y = int(mu["m10"] / (mu["m00"] + 1e-7) ), int(mu["m01"] / (mu["m00"] + 1e-7))
        # 重心を丸でくくる
        # cv.circle(img, (x, y), 4, 100, 2, 4)
        return x, y

    def reset(self):
        '''環境を初期化する'''
        print('Environment.reset')
        # xTODO: ロボットハンドがランダムに進路をとる
        Arduino().stop()  # 停止
        Arduino().release()  # 手を開く
        wheel = np.array([u'49', u'50', u'51', u'52'])  # 前 後 右 左
        for i in range(4):
            np.random.shuffle(wheel)
            droid.usbserialWrite((wheel[0].encode('utf-8')), uuid)
            time.sleep(8)

        # 赤色の面積, 重心の位置を取得する
        frame = self.cameraCapture('image_0')
        _, area_sum = self.calc_area(frame)
        area_v = 0
        center_pos_x, center_pos_y = self.calc_center(frame)
        center_v_x = 0
        center_v_y = 0

        observation = (area_sum, area_v, center_pos_x, center_pos_y, center_v_x, center_v_y)
        return observation, frame

    def get_env(self, area_sum_before, center_pos_before_x, center_pos_before_y):
        '''環境を認識する'''
        '''カメラで写真をとりOpenCVで面積と重心を取得する'''
        frame = self.cameraCapture('image_1')
        # 赤色の面積とその変化量, 重心の位置とその変化量を取得する
        _, area_sum = self.calc_area(frame)
        area_v = area_sum - area_sum_before
        center_pos_x, center_pos_y = self.calc_center(frame)
        center_v_x = center_pos_x - center_pos_before_x
        center_v_y = center_pos_y - center_pos_before_y
        # 観測量として返す
        observation = (area_sum, area_v, center_pos_x, center_pos_y, center_v_x, center_v_y)
        return observation, frame

    def act_env(self, observation, action):
        '''xTODO: 決定したactionに従って、ロボットハンドを動かす'''
        # Arduino().stop()  # はじめに停止させないとモーター動かない
        if action == 0:  # 前
            Arduino().go()
            # time.sleep(3)
        elif action == 1:  # 後
            Arduino().back()
            # time.sleep(3)
        elif action == 2:  # 右
            Arduino().right()
            # time.sleep(3)
        elif action == 3:  # 左
            Arduino().left()
            # time.sleep(3)
        elif action == 4:  # 握る
            Arduino().hold()
            # time.sleep(3)
        elif action == 5:  # 離す
            Arduino().release()
            # time.sleep(3)
        area_sum, _, center_pos_x, center_pos_y, _, _ = observation
        observation_next, _ = self.get_env(area_sum, center_pos_x, center_pos_y)
        done = self.is_done(observation_next)
        return observation_next, done

    def is_done(self, observation):
        '''observationによって終了判定をする'''
        #終了判定は面積が閾値以上&面積の変化なし（重心位置が画像の真ん中？カメラの位置によるけど，とりあえずはなし）
        done = False
        area_sum, area_v, center_pos_x, center_pos_y, center_v_x, center_v_y = observation
        if area_sum > AREA_THRESH and area_v < 1:
            done = True
        return done

    def run(self):
        '''実行'''
        print('Environment.run')
        complete_episodes = 0  # 18step以上連続で立ち続けた試行数
        is_episode_final = False  # 最終試行フラグ

        for episode in range(NUM_EPISODES):  # 試行数分繰り返す
            observation, frame = self.reset()  # 環境の初期化

            for step in range(MAX_STEPS):  # 1エピソードのループ
                print('Step: {0} of Episode: {1}'.format(step+1, episode))
                # 行動を求める
                action = self.agent.get_action(observation, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, done = self.act_env(observation, action)

                # 報酬を与える
                if done:  # ステップ数が1955経過するか、一定角度以上傾くとdoneはtrueになる
                    if step < MAX_STEPS*0.975:
                        reward = -1  # 途中でこけたら罰則として報酬-1を与える
                        complete_episodes = 0  # 4step以上連続で立ち続けた試行数をリセット
                    else:
                        reward = 1  # 立ったまま終了時は報酬1を与える
                        complete_episodes += 1  # 連続記録を更新
                else:
                    reward = 0  # 途中の報酬は0

                # step+1の状態observation_nextを用いて,Q関数を更新する
                self.agent.update_Q_function(observation, action, reward, observation_next)

                # 観測の更新
                observation = observation_next

                # 終了時の処理
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break

            if is_episode_final is True:  # 最終試行では動画を保存と描画
                break

            if complete_episodes >= 5:  # 5連続成功なら
                print('5回連続成功')
                is_episode_final = True  # 次の試行を描画を行う最終試行とする



# main
robot_hand_env = Environment()
robot_hand_env.run()
