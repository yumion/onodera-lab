import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_TPS.mp4', fourcc, 30.0, (640, 480))

while True:
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow("TPS", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
