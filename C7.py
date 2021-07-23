import cv2
import numpy as np
import pyperclip

# demo color mask isolation with car

def empty(a):
    lower, upper = getTrackbarsVals()
    pyperclip.copy(np.array2string(np.append(lower,upper), separator=', '))

def getTrackbarsVals():
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None,  scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2RGB)
        imageBlank = np.zeros((imgArray[0][0].shape[0], imgArray[0][0].shape[1], 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2RGB)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

path = 'Resources/lambo.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 7, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 128, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 249, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 123, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

img = cv2.imread(path)
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

while True:
    lower, upper = getTrackbarsVals()
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    imgStacked = stackImages(0.8,([img,imgHSV], [mask,imgResult]))
    cv2.imshow("r", imgStacked)

    if cv2.waitKey(1) != -1:
        cv2.destroyAllWindows()
        break
