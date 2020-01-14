# coding: utf-8
import cv2
import numpy as np
import serial
import time
import sys
import os

from utils.realsensecv import RealsenseCapture
from utils.utils import calc_center, green_detect

os.system(f'sudo chmod 666 {sys.argv[1]}')
ser = serial.Serial(port=sys.argv[1], baudrate=115200)
cap = RealsenseCapture()


def send_serial(motor, value, isreading=False):
    '''シリアル通信'''
    send = motor * 32 + value  # 8bitを10進数で表記
    send = send.to_bytes(1, 'big')  # byteに変換(ASCIIで表示される)
    ser.write(send)
    print(f"send: {send}, int: {int.from_bytes(send, 'big')}, bit: {format(int.from_bytes(send, 'big'), '08b')}")
    # read
    if isreading:
        while ser.inWaiting() > 0:
            read = ser.readline()
            read = read.strip().decode('utf-8')
            print(read)


CENTER_LINE = 423

cap.start()
time.sleep(5)

# default
params = [0, 0, 1, 0, 9]
for i, param in enumerate(params):
    send_serial(i, param, True)

ret, frames = cap.read(is_filtered=False)
start_rgb = frames[0]
start_depth = frames[1]
start_depth_pixels = (start_depth > 0).sum()

while True:
    ret, frames = cap.read(is_filtered=False)
    color_frame = frames[0]
    depth_frame = frames[1]

    mask = green_detect(color_frame.copy())
    center_pos_x, center_pos_y = calc_center(mask)
    # print(f'G({center_pos_x}, {center_pos_y})')
    target_distance = cap.depth_frame.get_distance(center_pos_x, center_pos_y)

    cv2.circle(color_frame, (center_pos_x, center_pos_y), 5, (0, 0, 255), thickness=-1)
    cv2.line(color_frame, (CENTER_LINE, 0), (CENTER_LINE, cap.HEGIHT), (255, 0, 0))
    images = np.hstack((color_frame, depth_frame))
    cv2.imshow('RealSense', images)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    vertical_pos = (0.0016 * target_distance * 100 - 0.0004) * (center_pos_x - CENTER_LINE)  # ピクセル間距離(cm)
    # print('vertical position: ', vertical_pos)
    vertical_deg = (max(min(vertical_pos // 0.0216, 90), -90) + 90) // 10  # 角度に変換して上限下限を制限して-90~90を0~18に変換

    depth_pixels = (depth_frame > 0).sum()
    print('Depth value: ', depth_pixels / start_depth_pixels)

    # Depth画像が真っ黒になるまで直進する
    if depth_pixels / start_depth_pixels < 0.1:
        '''ルールベースでつかむ'''
        params = [2, 2]  # 距離を詰める
        for i, param in enumerate(params):
            send_serial(i, param, True)
        time.sleep(3)  # 1.5cm
        print('reached')

        params = [0, 0, 0, 0]  # つかむ
        for i, param in enumerate(params):
            send_serial(i, param, True)
        print('grasp')
        time.sleep(2)

        send_serial(3, 1, True)  # 持ち上げる
        print('bring up')
        time.sleep(2)
        break
    else:
        params = [3, 3, 1, 0, int(vertical_deg)]
        for i, param in enumerate(params):
            send_serial(i, param, True)


for i in range(5):
    ret, frames = cap.read(is_filtered=False)
    color_frame = frames[0]
    depth_frame = frames[1]
    images = np.hstack((color_frame, depth_frame))
    cv2.imshow('RealSense', images)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

    depth_pixels = (depth_frame > 0).sum()
    print(f'count: {i} | {depth_pixels / start_depth_pixels}')

    if depth_pixels / start_depth_pixels < 0.5:
        # デプス画像の視界が開けなければ把持失敗
        print('Failed')
        break
    if i == 4:
        print('Success!')
        # send_serial(3, 0, True)
        # time.sleep(3)

"""
# ホームポジションへ戻る
from cv2 import aruco
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
MAX_SPEED = 20
GOAL_POS = cap.WIDTH // 2
while True:
    ret, frames = cap.read()
    color_frame = frames[0]
    depth_frame = frames[1]
    # マーカーを検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(color_frame, dictionary)
    if ids is not None:
        target_pos = [int(x) for x in corners[0][0][0]]
        print('id: ', ids[0])
        print('corners: ', target_pos)
        target_distance = cap.depth_frame.get_distance(target_pos[0], target_pos[1])
        GOAL_POS = int(cap.WIDTH // 2 + 3.2 / (0.0016 * target_distance * 100 + 0.0006))  # カメラ中心からロボット中心に合わせる
        error_distance = (target_pos[0] - GOAL_POS) / GOAL_POS
        print(f'error: {error_distance:.3f} ({(0.0016 * target_distance * 100 + 0.0006) * abs(target_pos[0] - GOAL_POS):.2f}cm) |   target distance: {target_distance * 100:.2f}cm')

        r_motor = (1 - error_distance) / 2 * MAX_SPEED
        l_motor = (1 + error_distance) / 2 * MAX_SPEED

    else:
        r_motor = 0
        l_motor = 20

    params = [int(r_motor), int(l_motor)]
    for i, param in enumerate(params):
        send_serial(i, param, True)

    aruco.drawDetectedMarkers(color_frame, corners, ids)  # マーカーを四角で囲む
    cv2.line(color_frame, (GOAL_POS, 0), (GOAL_POS, cap.HEGIHT), (255, 0, 0))
    images = np.hstack((color_frame, depth_frame))
    cv2.imshow('RealSense', images)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    if 0 < target_distance < 0.17:
        print('reached!')
        send_serial(0, 0)
        send_serial(1, 0)
        break


send_serial(3, 0, True)  # 腕を下げる
time.sleep(3)


params = [0, 0, 1, 0, 9]
for i, param in enumerate(params):
    send_serial(i, param, True)

ser.close()
cap.release()
cv2.destroyAllWindows()
"""
