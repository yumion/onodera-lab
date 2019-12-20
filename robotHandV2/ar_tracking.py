import time
import sys
import numpy as np
import cv2
from cv2 import aruco

import serial
ser = serial.Serial(port=sys.argv[1], baudrate=115200)
# RealSense
from utils.realsensecv import RealsenseCapture
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


# ARマーカーの辞書
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

cap.start()
time.sleep(5)

MAX_SPEED = 20
GOAL_POS = cap.WIDTH // 2
r_motor = 0
l_motor = 0
error_distance = 0
target_distance = 0

# default
params = [r_motor, l_motor, 0, 1, 9]
for i, param in enumerate(params):
    send_serial(i, param, True)

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
        r_motor = 20
        l_motor = 1

    params = [int(r_motor), int(l_motor)]
    for i, param in enumerate(params):
        send_serial(i, param, True)

    aruco.drawDetectedMarkers(color_frame, corners, ids)  # マーカーを四角で囲む
    cv2.line(color_frame, (GOAL_POS, 0), (GOAL_POS, cap.HEGIHT), (255, 0, 0))
    images = np.hstack((color_frame, depth_frame))
    cv2.imshow('RGB', images)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    if 0 < target_distance < 0.17:
        print('reached!')
        send_serial(0, 0)
        send_serial(1, 0)
        break


send_serial(3, 0, True)  # 腕を下げる
time.sleep(3)
params = [0, 0, 1, 0, 9]  # 手を開く
for i, param in enumerate(params):
    send_serial(i, param, True)

ser.close()
cap.release()
cv2.destroyAllWindows()
