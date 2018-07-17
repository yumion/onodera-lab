import androidhelper as android
import sys, os
import time
import cv

droid = android.Android()


droid.cameraCapturePicture("/storage/7E9B-5A00/Picture/sampleImage.jpg")


cascPath = sys.argv[1]
faceCascade = cv.CascadeClassifier(cascPath)

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle( image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # show the frame
    cv.imshow("Frame", image)
    key = cv.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
