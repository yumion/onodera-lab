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
# 俯瞰カメラ
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FPS, 30)

# 動画を保存
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output_FPS.mp4', fourcc, 5.0, (640 * 2, 480))
out2 = cv2.VideoWriter('output_TPS.mp4', fourcc, 5.0, (640, 480))


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


cap.start()
time.sleep(5)


MAX_SPEED = 20
# GOAL_POS = cap.WIDTH // 2
GOAL_POS = 423
r_motor = 0
l_motor = 0

# default
params = [r_motor, l_motor, 0, 0, 9]
for i, param in enumerate(params):
    send_serial(i, param, True)

while True:
    # 俯瞰カメラ
    ret2, frame2 = cap2.read()
    # 主観カメラ
    ret, frames = cap.read(is_filtered=False)
    color_frame = frames[0]
    depth_frame = frames[1]

    mask = green_detect(color_frame.copy())
    mask_pixels = (mask > 0).sum()
    print('Area: ', mask_pixels / (mask.shape[0] * mask.shape[1]))
    center_pos_x, center_pos_y = calc_center(mask)
    print(f'G({center_pos_x}, {center_pos_y})')

    target_distance = cap.depth_frame.get_distance(center_pos_x, center_pos_y)
    # GOAL_POS = int(cap.WIDTH // 2 + 3.2 / (0.0016 * target_distance * 100 + 0.0006))  # カメラ中心からロボット中心に合わせる
    error_distance = (center_pos_x - GOAL_POS) / GOAL_POS
    print(f'error: {error_distance} ({(0.0016 * target_distance * 100 + 0.0006) * abs(center_pos_x - GOAL_POS)}cm) |   target distance: {target_distance * 100}cm')

    r_motor = (1 - error_distance) / 2 * MAX_SPEED
    l_motor = (1 + error_distance) / 2 * MAX_SPEED

    params = [int(r_motor), int(l_motor)]
    for i, param in enumerate(params):
        send_serial(i, param, True)

    mask_RGB = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.circle(mask_RGB, (center_pos_x, center_pos_y), 5, (0, 0, 255), thickness=-1)
    cv2.line(mask_RGB, (GOAL_POS, 0), (GOAL_POS, cap.HEGIHT), (255, 0, 0))
    images = np.hstack((color_frame, mask_RGB))
    out1.write(images)  # 動画を保存
    out2.write(frame2)  # 動画を保存
    cv2.imshow('RGB', np.hstack((frame2, images)))
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    if 0 < target_distance < 0.16 or mask_pixels / (mask.shape[0] * mask.shape[1]) > 0.3:
        print('reached!')
        break

    # time.sleep(0.2)

send_serial(0, 0)
send_serial(1, 0)
ser.close()
cap.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
