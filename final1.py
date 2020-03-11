#LIBRARY NECESSARY
import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils

colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


color1 = 0
color2 = 0
color3 = 0

ranges = 20
set_color = False
step = 0


def nothing(x):
    global color1, color2, color3
    global lower_blueA1, lower_blueA2, lower_blueA3
    global upper_blueA1, upper_blueA2, upper_blueA3
    global lower_blueB1, lower_blueB2, lower_blueB3
    global upper_blueB1, upper_blueB2, upper_blueB3
    global lower_blueC1, lower_blueC2, lower_blueC3
    global upper_blueC1, upper_blueC2, upper_blueC3

    saturation_th1 = cv.getTrackbarPos('saturation_th1', 'img_result')
    value_th1 = cv.getTrackbarPos('value_th1', 'img_result')

    saturation_th2 = cv.getTrackbarPos('saturation_th2', 'img_result')
    value_th2 = cv.getTrackbarPos('value_th2', 'img_result')

    saturation_th3 = cv.getTrackbarPos('saturation_th3', 'img_result')
    value_th3 = cv.getTrackbarPos('value_th3', 'img_result')

    color1 = int(color1)
    color2 = int(color2)
    color3 = int(color3)

    # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
    if color1 < ranges:
        lower_blueA1 = np.array([color1 - ranges + 180, saturation_th1, value_th1])
        upper_blueA1 = np.array([180, 255, 255])
        lower_blueA2 = np.array([0, saturation_th1, value_th1])
        upper_blueA2 = np.array([color1, 255, 255])
        lower_blueA3 = np.array([color1, saturation_th1, value_th1])
        upper_blueA3 = np.array([color1 + ranges, 255, 255])
        #     print(i-range+180, 180, 0, i)
        #     print(i, i+range)

    elif color1 > 180 - ranges:
        lower_blueA1 = np.array([color1, saturation_th1, value_th1])
        upper_blueA1 = np.array([180, 255, 255])
        lower_blueA2 = np.array([0, saturation_th1, value_th1])
        upper_blueA2 = np.array([color1 + ranges - 180, 255, 255])
        lower_blueA3 = np.array([color1 - ranges, saturation_th1, value_th1])
        upper_blueA3 = np.array([color1, 255, 255])
        #     print(i, 180, 0, i+range-180)
        #     print(i-range, i)
    else:
        lower_blueA1 = np.array([color1, saturation_th1, value_th1])
        upper_blueA1 = np.array([color1 + ranges, 255, 255])
        lower_blueA2 = np.array([color1 - ranges, saturation_th1, value_th1])
        upper_blueA2 = np.array([color1, 255, 255])
        lower_blueA3 = np.array([color1 - ranges, saturation_th1, value_th1])
        upper_blueA3 = np.array([color1, 255, 255])
        #     print(i, i+range)
        #     print(i-range, i)


    if color2 < ranges:
        lower_blueB1 = np.array([color2 - ranges + 180, saturation_th2, value_th2])
        upper_blueB1 = np.array([180, 255, 255])
        lower_blueB2 = np.array([0, saturation_th2, value_th2])
        upper_blueB2 = np.array([color2, 255, 255])
        lower_blueB3 = np.array([color2, saturation_th2, value_th2])
        upper_blueB3 = np.array([color2 + ranges, 255, 255])
        #     print(i-range+180, 180, 0, i)
        #     print(i, i+range)

    elif color2 > 180 - ranges:
        lower_blueB1 = np.array([color2, saturation_th2, value_th2])
        upper_blueB1 = np.array([180, 255, 255])
        lower_blueB2 = np.array([0, saturation_th2, value_th2])
        upper_blueB2 = np.array([color2 + ranges - 180, 255, 255])
        lower_blueB3 = np.array([color2 - ranges, saturation_th2, value_th2])
        upper_blueB3 = np.array([color2, 255, 255])
        #     print(i, 180, 0, i+range-180)
        #     print(i-range, i)
    else:
        lower_blueB1 = np.array([color2, saturation_th2, value_th2])
        upper_blueB1 = np.array([color2 + ranges, 255, 255])
        lower_blueB2 = np.array([color2 - ranges, saturation_th2, value_th2])
        upper_blueB2 = np.array([color2, 255, 255])
        lower_blueB3 = np.array([color2 - ranges, saturation_th2, value_th2])
        upper_blueB3 = np.array([color2, 255, 255])
        #     print(i, i+range)
        #     print(i-range, i)

    if color3 < ranges:
        lower_blueC1 = np.array([color3 - ranges + 180, saturation_th3, value_th3])
        upper_blueC1 = np.array([180, 255, 255])
        lower_blueC2 = np.array([0, saturation_th3, value_th3])
        upper_blueC2 = np.array([color3, 255, 255])
        lower_blueC3 = np.array([color3, saturation_th3, value_th3])
        upper_blueC3 = np.array([color3 + ranges, 255, 255])
        #     print(i-range+180, 180, 0, i)
        #     print(i, i+range)

    elif color3 > 180 - ranges:
        lower_blueC1 = np.array([color3, saturation_th3, value_th3])
        upper_blueC1 = np.array([180, 255, 255])
        lower_blueC2 = np.array([0, saturation_th3, value_th3])
        upper_blueC2 = np.array([color3 + ranges - 180, 255, 255])
        lower_blueC3 = np.array([color3 - ranges, saturation_th3, value_th3])
        upper_blueC3 = np.array([color3, 255, 255])
        #     print(i, 180, 0, i+range-180)
        #     print(i-range, i)
    else:
        lower_blueC1 = np.array([color3, saturation_th3, value_th3])
        upper_blueC1 = np.array([color3 + ranges, 255, 255])
        lower_blueC2 = np.array([color3 - ranges, saturation_th3, value_th3])
        upper_blueC2 = np.array([color3, 255, 255])
        lower_blueC3 = np.array([color3 - ranges, saturation_th3, value_th3])
        upper_blueC3 = np.array([color3, 255, 255])
        #     print(i, i+range)
        #     print(i-range, i)


cv.namedWindow('img_color')
cv.namedWindow('img_result')

cv.createTrackbar('saturation_th1', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('saturation_th1', 'img_result', 130)
cv.createTrackbar('value_th1', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('value_th1', 'img_result', 130)

cv.createTrackbar('saturation_th2', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('saturation_th2', 'img_result', 130)
cv.createTrackbar('value_th2', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('value_th2', 'img_result', 130)

cv.createTrackbar('saturation_th3', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('saturation_th3', 'img_result', 130)
cv.createTrackbar('value_th3', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('value_th3', 'img_result', 130)

cap = cv.VideoCapture(0)


while(True):

    ret,img_color = cap.read()
    img_color = cv.flip(img_color, 1)

    if ret == False:
        continue;


    img_color2 = img_color.copy()
    img_hsv = cv.cvtColor(img_color2, cv.COLOR_BGR2HSV)

    height, width = img_color.shape[:2]
    cx = int(width / 2)
    cy = int(height / 2)


    if set_color == False:

        rectangle_color = (0, 255, 0)

        if step == 1:
            rectangle_color = (255, 0, 0)

        elif step == 2:
            rectangle_color = (0, 0, 255)

        cv.rectangle(img_color, (cx - 20, cy - 20), (cx + 20, cy + 20), rectangle_color, 5)


    else:

        # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
        img_maskA1 = cv.inRange(img_hsv, lower_blueA1, upper_blueA1)
        img_maskA2 = cv.inRange(img_hsv, lower_blueA2, upper_blueA2)
        img_maskA3 = cv.inRange(img_hsv, lower_blueA3, upper_blueA3)
        temp = cv.bitwise_or(img_maskA1, img_maskA2)
        img_maskA = cv.bitwise_or(img_maskA3, temp)

        img_maskB1 = cv.inRange(img_hsv, lower_blueB1, upper_blueB1)
        img_maskB2 = cv.inRange(img_hsv, lower_blueB2, upper_blueB2)
        img_maskB3 = cv.inRange(img_hsv, lower_blueB3, upper_blueB3)
        temp = cv.bitwise_or(img_maskB1, img_maskB2)
        img_maskB = cv.bitwise_or(temp, img_maskB3)

        img_maskC1 = cv.inRange(img_hsv, lower_blueC1, upper_blueC1)
        img_maskC2 = cv.inRange(img_hsv, lower_blueC2, upper_blueC2)
        img_maskC3 = cv.inRange(img_hsv, lower_blueC3, upper_blueC3)
        temp = cv.bitwise_or(img_maskC1, img_maskC2)
        img_maskC = cv.bitwise_or(img_maskC3, temp)

        # 모폴로지 연산
        kernel = np.ones((11,11), np.uint8)
        img_maskA = cv.morphologyEx(img_maskA, cv.MORPH_OPEN, kernel)
        img_maskA = cv.morphologyEx(img_maskA, cv.MORPH_CLOSE, kernel)


        kernel = np.ones((11,11), np.uint8)
        img_maskB = cv.morphologyEx(img_maskB, cv.MORPH_OPEN, kernel)
        img_maskB = cv.morphologyEx(img_maskB, cv.MORPH_CLOSE, kernel)

        kernel = np.ones((11, 11), np.uint8)
        img_maskC = cv.morphologyEx(img_maskC, cv.MORPH_OPEN, kernel)
        img_maskC = cv.morphologyEx(img_maskC, cv.MORPH_CLOSE, kernel)


        # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
        temp = cv.bitwise_or(img_maskA, img_maskB)
        img_maskD = cv.bitwise_or(temp, img_maskC)
        img_result = cv.bitwise_and(img_color, img_color, mask=img_maskD)

        # center values
        centerX1, centerY1, centerX2, centerY2, centerX3, centerY3 = 0, 0, 0, 0, 0, 0
        ref_width = 0


        # 라벨링
        numOfLabelsA, img_labelA, statsA, centroidsA = cv.connectedComponentsWithStats(img_maskA)

        for idx, centroid in enumerate(centroidsA):
            if statsA[idx][0] == 0 and statsA[idx][1] == 0:
                continue

            if np.any(np.isnan(centroid)):
                continue

            x, y, width, height, area = statsA[idx]
            centerX1, centerY2 = int(centroid[0]), int(centroid[1])


            if area > 1500:
                cv.circle(img_color, (centerX1, centerY2), 10, (0,0,255), 10)
                cv.rectangle(img_color, (x,y), (x+width,y+height), (0,0,255))


        numOfLabelsB, img_labelB, statsB, centroidsB = cv.connectedComponentsWithStats(img_maskB)

        for idx, centroid in enumerate(centroidsB):
            if statsB[idx][0] == 0 and statsB[idx][1] == 0:
                continue

            if np.any(np.isnan(centroid)):
                continue

            x, y, width,height,area = statsB[idx]
            centerX2,centerY2 = int(centroid[0]), int(centroid[1])


            if area > 1500:
                cv.circle(img_color, (centerX2, centerY2), 10, (0,255,0), 10)
                cv.rectangle(img_color, (x,y), (x+width,y+height), (0, 255 ,0))


        numOfLabelsC, img_labelC, statsC, centroidsC = cv.connectedComponentsWithStats(img_maskC)
        for idx, centroid in enumerate(centroidsC):
            if statsC[idx][0] == 0 and statsC[idx][1] == 0:
                continue

            if np.any(np.isnan(centroid)):
                continue

            x, y, ref_width, height, area = statsC[idx]
            centerX3, centerY3 = int(centroid[0]), int(centroid[1])

            if area > 1500:
                cv.circle(img_color, (centerX3, centerY3), 10, (255, 0, 0), 10)
                cv.rectangle(img_color, (x, y), (x + ref_width, y + height), (255, 0, 0))


        # cv.line(img_color, (centerX1, centerY1), (centerX3, centerY3), colors[1], 2)
        # cv.line(img_color, (centerX2, centerY2), (centerX3, centerY3), colors[2], 2)

        # cv.line(img_color, (centerX1, centerY1), (centerX3, centerY3), colors[0], 2)
        # cv.line(img_color, (centerX2, centerY2), (centerX3, centerY3), colors[1], 2)

        ####distance measuring#####
        # gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
        # gray = cv.GaussianBlur(gray, (7, 7), 0)
        #
        # # perform edge detection, then perform a dilation + erosion to
        # # close gaps in between object edges
        # edged = cv.Canny(gray, 50, 100)
        # edged = cv.dilate(edged, None, iterations=1)
        # edged = cv.erode(edged, None, iterations=1)

        # # find contours in the edge map
        # cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        #
        # (cnt, _) = contours.sort_contours(cnts)

        # actual_length_of_ref_object = 0.999; #needs to be measured from actual environment
        #
        #
        # pixels_per_metics = ref_width/actual_length_of_ref_object
        # D1 = dist.euclidean((centerX1, centerY1), (centerX3, centerY3))/ pixels_per_metics
        # (mX1, mY1) = midpoint((centerX1, centerY1), (centerX3, centerY3))
        # cv.putText(img_color, "{:.1f}in".format(D1), (int(mX1), int(mY1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.55, colors[3], 2)
        #
        # D2 = dist.euclidean((centerX2, centerY2), (centerX3, centerY3)) / pixels_per_metics
        # (mX2, mY2) = midpoint((centerX2, centerY2), (centerX3, centerY3))
        # cv.putText(img_color, "{:.1f}in".format(D2), (int(mX2), int(mY2 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.55, colors[4], 2)
        #

        cv.imshow('img_result', img_result)


    cv.imshow('img_color', img_color)


    key = cv.waitKey(1) & 0xFF

    if key == 27: # esc
        break

    elif key == 32: # space
        if step == 0:
            roi = img_color2[cy-20:cy+20, cx-20:cx+20]
            roi = cv.medianBlur(roi, 5)
            cv.imshow("roi1", roi)
            hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            h,s,v = cv.split(hsv)
            color1 = h.mean()
            print(color1)
            step += 1

        elif step == 1:
            roi = img_color2[cy-20:cy+20, cx-20:cx+20]
            roi = cv.medianBlur(roi, 5)
            cv.imshow("roi2", roi)
            hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            h,s,v = cv.split(hsv)
            color2 = h.mean()
            print(color2)
            step += 1

        elif step == 2:
            roi = img_color2[cy - 20:cy + 20, cx - 20:cx + 20]
            roi = cv.medianBlur(roi, 5)
            cv.imshow("roi3", roi)
            hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            color3 = h.mean()
            set_color = True
            nothing(0)
            print(color3)
            step += 1


cap.release()
cv.destroyAllWindows()