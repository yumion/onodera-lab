import numpy as np
import matplotlib.pyplot as plt
import cv2


# 赤色のマスク
def red_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 赤色のHSVの値域1
    hsv_min = np.array([0,127,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)
    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    return mask1 + mask2

# 青色のマスク
def blue_detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([80, 150, 0])
    hsv_max = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

# 黒色のマスク
def black_detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([0, 0, 0])
    hsv_max = np.array([179, 128, 100])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask

# 面積計算
def calc_area(img):
    pix_area = cv2.countNonZero(img) #ピクセル数
    # パーセントを算出
    h, w = img.shape #frameの面積
    per = round(100*float(pix_area)/(w * h), 3) #比率

    return pix_area, per

# 重心を求める
def calc_center(img):
    mu = cv2.moments(img, False)
    x, y = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
    #重心を丸でくくる
    cv2.circle(img, (x,y), 4, 100, 2, 4)

    return x, y


cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号，外付けUSBカメラだと1番

while True:
    # retは画像を取得成功フラグ(Boolian)，frameは動画の1フレームを切り出し
    ret, frame = cap.read()
    # 指定した色でセグメントを塗りつぶす(マスクをかぶせる)
    mask = red_detect(frame) #red,blue,black

    # ピクセル数を計算
    pix_area, per = calc_area(mask)
    # 面積をテキストで表示
    cv2.putText(mask, "Moment[px]: " + str(pix_area), (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(mask, "Percent[%]: " + str(per), (0, 120), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)

    #重心を求める
    x, y = calc_center(mask)
    #重心の座標を画面に表示
    cv2.putText(mask, "G({},{})".format(x, y), (0, 200), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)

    #差分を取る
    frame2 = cap.read()[1]
    mask2 = red_detect(frame2)
    _, per2 = calc_area(mask2)
    x2, y2 = calc_center(mask2)

    diff_area = abs(per2 - per)
    diff_center = np.array([x2 - x, y2 - y])

    print('diff_area: {0:.3f}'.format(diff_area))
    print('diff_center({})'.format(diff_center))

    # フレームを表示する
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
