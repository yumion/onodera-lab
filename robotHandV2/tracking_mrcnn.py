# coding: utf-8
import time
import numpy as np
import cv2
from mrcnn import visualize

import sys
import serial
ser = serial.Serial(port=sys.argv[1], baudrate=115200)
# RealSense
from utils.realsensecv import RealsenseCapture
cap = RealsenseCapture()
# Mask R-CNN
from utils.inference_mrcnn import Inference_model
model = Inference_model()

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# filtered_classNames = ['BG', 'bottle', 'cup', 'banana', 'orange', 'remote', 'cell phone']


def calc_center(img):
    '''重心座標(x,y)を求める'''
    mu = cv2.moments(img, False)
    x = int(mu["m10"] / (mu["m00"] + 1e-7))
    y = int(mu["m01"] / (mu["m00"] + 1e-7))
    return x, y


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


MAX_SPEED = 20
GOAL_POS = 423
r_motor = 0
l_motor = 0

cap.start()
# time.sleep(5)

while True:
    start_time = time.time()
    ret, images = cap.read()
    rgb_image = images[0]
    depth_image = images[1]
    # Run detection
    result = model.detect([rgb_image], verbose=1)[0]
    N = result['rois'].shape[0]  # 検出数
    result_image = rgb_image.copy()
    colors = visualize.random_colors(N)
    for i in range(N):
        '''クラス関係なく1物体ごと処理を行う'''
        if class_names[result['class_ids'][i]] in sys.argv[2]:
            # Color
            color = colors[i]
            rgb = (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Bbox
            result_image = visualize.draw_box(result_image, result['rois'][i], rgb)
            # Class & Score
            text_top = class_names[result['class_ids'][i]] + ':' + str(result['scores'][i])
            result_image = cv2.putText(result_image, text_top,
                                       (result['rois'][i][1], result['rois'][i][0]),
                                       font, 0.7, rgb, 1, cv2.LINE_AA)
            # Mask
            mask = result['masks'][:, :, i]
            result_image = visualize.apply_mask(result_image, mask, color)
            # Distance
            mask_binary = mask.astype('uint8')

            mask_pixels = (mask_binary > 0).sum()
            print('Area: ', mask_pixels / (mask_binary.shape[0] * mask_binary.shape[1]))

            center_pos = calc_center(mask_binary)
            print(f'G{center_pos}')
            target_distance = cap.depth_frame.get_distance(center_pos[0], center_pos[1])
            text_bottom = '{:.3f}m'.format(target_distance)

            result_image = cv2.putText(result_image, text_bottom,
                                       (result['rois'][i][1], result['rois'][i][0] - 15),
                                       font, 0.7, rgb, 1, cv2.LINE_AA)
            # log
            print('class: {} | Score: {} | Distance: {}m'.format(class_names[result['class_ids'][i]], result['scores'][i], target_distance))

    print('FPS:', 1 / (time.time() - start_time))

    # control dc motor
    error_distance = (center_pos[0] - GOAL_POS) / GOAL_POS
    print(f'error: {error_distance} ({(0.0016 * target_distance * 100 + 0.0006) * abs(center_pos[0] - GOAL_POS)}cm) |   target distance: {target_distance * 100}cm')

    r_motor = (1 - error_distance) / 2 * MAX_SPEED
    l_motor = (1 + error_distance) / 2 * MAX_SPEED
    params = [int(r_motor), int(l_motor)]
    # for i, param in enumerate(params):
    #     send_serial(i, param, True)

    # cv2.circle(result_image, (center_pos[0], center_pos[1]), 5, (0, 0, 255), thickness=-1)
    cv2.line(result_image, (GOAL_POS, 0), (GOAL_POS, cap.HEGIHT), (255, 0, 0))
    cv2.imshow('Mask R-CNN', np.hstack((result_image, depth_image)))

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    if 0 < target_distance < 0.16 or mask_pixels / (mask.shape[0] * mask.shape[1]) > 0.3:
        print('reached!')
        break
