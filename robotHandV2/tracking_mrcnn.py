# coding: utf-8
import time
import sys
# argv[1]: port, argv[2]: class
import numpy as np
import cv2
from utils.utils import calc_center
from ncc.video.utils import FPS
# to Arduino
import serial
import os
os.system(f'sudo chmod 666 {sys.argv[1]}')
ser = serial.Serial(port=sys.argv[1], baudrate=115200)
# RealSense
from utils.realsensecv import RealsenseCapture
cap = RealsenseCapture()
# Mask R-CNN
from utils.inference_mrcnn import Inference_model, render
model = Inference_model()

# filtered_classNames = ['BG', 'bottle', 'cup', 'banana', 'orange', 'remote', 'cell phone']


# 動画を保存
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_tracking.mp4', fourcc, 5.0, (640 * 2, 480))


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
fps = FPS()
# time.sleep(2)

MAX_SPEED = 20
GOAL_POS = cap.WIDTH // 2
r_motor = 0
l_motor = 0

# default
params = [r_motor, l_motor, 1, 0, 9]
for i, param in enumerate(params):
    send_serial(i, param, True)

while True:
    # 主観カメラ
    ret, images = cap.read()
    rgb_image = images[0]
    depth_image = images[1]
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    params = [0, 0]
    for i, param in enumerate(params):
        send_serial(i, param, True)
    # Run detection
    padding_image = np.zeros((960, 1280, 3), np.uint8)
    padding_image[240:720, 320:960] = rgb_image  # 近距離でも遠くに見えるようにパディングする
    result = model.detect([padding_image], verbose=0)[0]
    result_image, mask = render(result, padding_image.copy(), sys.argv[2])
    result_image = result_image[240:720, 320:960]  # paddingを元に戻す
    if mask is not None:
        mask = mask[240:720, 320:960]
    # fps.calculate(result_image)  # FPSを測定
    if mask is None:
        cv2.line(result_image, (GOAL_POS, 0), (GOAL_POS, cap.HEGIHT), (255, 0, 0))
        cv2.imshow('Mask R-CNN', np.hstack((result_image, depth_image)))
        send_serial(1, MAX_SPEED, True)
        continue

    # Distance
    mask_binary = mask.astype('uint8')
    mask_pixels = (mask_binary > 0).sum()
    print('Area: ', mask_pixels / (mask_binary.shape[0] * mask_binary.shape[1]))

    center_pos = calc_center(mask_binary)
    print(f'G{center_pos}')
    target_distance = cap.depth_frame.get_distance(center_pos[0], center_pos[1])

    # control dc motor
    # GOAL_POS = int(cap.WIDTH // 2 + 3.2 / (0.0016 * target_distance * 100 + 0.0006))  # カメラ中心からロボット中心に合わせる
    GOAL_POS = 423
    # print(GOAL_POS)
    error_distance = (center_pos[0] - GOAL_POS) / GOAL_POS
    print(f'error: {error_distance:.4f} ({(0.0016 * target_distance * 100 + 0.0006) * abs(center_pos[0] - GOAL_POS):.3f}cm) |   target distance: {target_distance * 100:.3f}cm')

    r_motor = (1 - error_distance) / 2 * MAX_SPEED
    l_motor = (1 + error_distance) / 2 * MAX_SPEED
    params = [int(r_motor), int(l_motor)]
    for i, param in enumerate(params):
        send_serial(i, param, True)

    cv2.circle(result_image, (center_pos[0], center_pos[1]), 5, (0, 0, 255), thickness=-1)
    cv2.line(result_image, (GOAL_POS, 0), (GOAL_POS, cap.HEGIHT), (255, 0, 0))
    out.write(np.hstack((result_image, depth_image)))  # 動画を保存
    cv2.imshow('Mask R-CNN', np.hstack((result_image, depth_image)))

    if 0 < target_distance < 0.24:
        print('reached!')
        break

send_serial(0, 0)
send_serial(1, 0)
ser.close()
cap.release()
out.release()
cv2.destroyAllWindows()
