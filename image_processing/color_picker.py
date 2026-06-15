import numpy as np
import cv2

# main function-----------------------------------------------------------------------------
def main():
    global img, cache, bar

    # read image
    img = cv2.imread("./img/RGBHSV.bmp")
    cache = img.copy()
    bar = []

    # display image
    cv2.imshow("target image", img)
    cv2.setMouseCallback("target image", mouse_event)

    # keep all windows until "ESC" button is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# create HSV-gradation image window function-------------------------------------------------
def createGradationImage(hue, sat, val):
    cimg = np.zeros((256, 256, 3), np.uint8) # initialize hsv gradation image with 0
    posHSV = [0, 0] # to show HSV value of clicked area

    for j in range(256):
        for i in range(256):
            cimg[j,i,0] = np.uint8(hue) # Hue
            cimg[j,i,1] = i             # Saturation
            cimg[j,i,2] = j             # Value

            # show HSV value area of click position
            if j == np.uint8(val) and i == np.uint8(sat):
                posHSV = [i, j]

    # put text on Image
    cv2.putText(cimg, "> satulation", (5, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
    cv2.putText(cimg, "|128", (128, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0,255))
    cv2.putText(cimg, "V vlue", (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0,255))
    cv2.putText(cimg, "- 128", (0, 131), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))

    # show HSV value of clicked area
    cv2.rectangle(cimg, (posHSV[0] - 2, posHSV[1] - 2), (posHSV[0] + 2, posHSV[1] + 2), (0, 255, 255), 1)

    cv2.imshow("color range", cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR))

# trackbar event function-------------------------------------------------------------------
def changeTrackbarRange(val):
    global av_s, av_v
    # update gradation image
    createGradationImage(val, av_s, av_v)

# mouse event function----------------------------------------------------------------------
def mouse_event(event, x, y, flg, prm):
    global img, cache, bar
    global av_h, av_s, av_v

    # when mouse is moved
    if event == cv2.EVENT_MOUSEMOVE:
        # --clear image (keep mark of the latest clicked area)
        mvcache = cache.copy()

        # show mouse position
        cv2.rectangle(mvcache, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), 1)
        cv2.imshow("target image", mvcache)

    # when left button is clicked
    elif event == cv2.EVENT_LBUTTONDOWN:
        # --clear image (to target image)
        cache = img.copy()

        print("-- BGR <-> HSV -----------------------------------------------------------")

        # average of each color components around pointing area
        av_b = np.mean(img[y - 2:y + 2, x - 2:x + 2, 0])
        av_g = np.mean(img[y - 2:y + 2, x - 2:x + 2, 1])
        av_r = np.mean(img[y - 2:y + 2, x - 2:x + 2, 2])

        print("(B,G,R) = (" + str(av_b) + ", " + str(av_g) + ", " + str(av_r) + ")")

        # average of HSV components around pointing area
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        av_h = np.mean(hsv[y - 2:y + 2, x - 2:x + 2, 0])
        av_s = np.mean(hsv[y - 2:y + 2, x - 2:x + 2, 1])
        av_v = np.mean(hsv[y - 2:y + 2, x - 2:x + 2, 2])

        print("(H,S,V) = (" + str(av_h) + ", " + str(av_s) + ", " + str(av_v) + ")")
        createGradationImage(av_h, av_s, av_v)

        # track bar
        cv2.namedWindow("color range", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        if bar == []:
            bar = cv2.createTrackbar("Hue", "color range", int(av_h), 179, changeTrackbarRange) # Max value of hue in openCV is 179 (RED)
        else:
            cv2.setTrackbarPos("Hue", "color range", int(av_h))

        # show clicked area
        cv2.rectangle(cache, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), 1)
        cv2.imshow("target image", cache)

# run---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
