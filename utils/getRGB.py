# coding: utf-8
import cv2
import numpy as np
import csv
import sys
import os


click_rgb = []  # クリックした座標のRGB

# マウスイベント時に処理を行う


def mouse_event(event, x, y, flags, param):
    image = param
    # 左クリックで座標を返す
    if event == cv2.EVENT_LBUTTONUP:
        click_rgb.append(image[x][y][::-1])
        print(image[x][y][::-1])  # RGB


img = cv2.imread(sys.argv[1])
cv2.namedWindow(sys.argv[1], cv2.WINDOW_AUTOSIZE)
# マウスイベント時に関数mouse_eventの処理を行う
cv2.setMouseCallback(sys.argv[1], mouse_event, img)

while True:
    cv2.imshow(sys.argv[1], img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

with open(sys.argv[2] + '.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['R', 'G', 'B'])
    writer.writerows(click_rgb)
