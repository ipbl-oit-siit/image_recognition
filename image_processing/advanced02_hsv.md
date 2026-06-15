# Image processing basics for static images

[back to the top page](../README.md)

---
### :red_square: Color space (HSV color space)

* Color can also be represented by HSV color space instead of RGB.
* **HSV** stands for **Hue**, **Saturation**, and **Value**.
* **Hue**: Color type (e.g., Red, Yellow, Green, Blue) represented by angle ($0$ to $360^\circ$).
* **Saturation**: Vividness of color.
* **Value**: Brightness of color.



#### Features of HSV color space

* It is **reversibly convertible** with RGB.
* It allows for more **intuitive color handling**, making it easy to specify areas like "a range of yellowish colors".

#### Data Range in OpenCV

When using OpenCV (`cv2`), the data ranges are scaled to fit within 8-bit integer values ($0$ to $255$):

* **Hue**: $0$ to $179$ (The actual angle $0$ to $360^\circ$ is divided by 2. For example, Hue = $38$ represents $38 \times 2 = 76^\circ$).
* **Saturation**: $0$ to $255$ ($0$ to $100\%$).
* **Value**: $0$ to $255$ ($0$ to $100\%$).

#### :blue_square: Color conversion with `cv2`

* You can convert a BGR image to an HSV image using the following function:
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

```



#### :o:Practice[HSV color space]

* Save the following sample code as a python file, and execute it. (`C:/oit/py25en/source/sample_hsv.py`)
* `sample_hsv.py`
```python
import cv2
import numpy as np

# Read image file
img = cv2.imread('./img/standard/Mandrill.bmp')
if img is None:
    print('ERROR: image file is not opened.')
    exit(1)

# Convert BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split into H, S, V channels
h, s, v = cv2.split(hsv)

# Show each channel as a grayscale image
cv2.imshow('Original', img)
cv2.imshow('Hue channel', h)
cv2.imshow('Saturation channel', s)
cv2.imshow('Value channel', v)

cv2.waitKey(0)
cv2.destroyAllWindows()

```


* It is O.K., if the windows pop up and you can observe how the components of Hue, Saturation, and Value are separated.

### :red_square: Advanced Application: Color Extraction using HSV

By analyzing pixel values in the HSV color space, we can define a specific range of values to extract target objects from an image.

#### :o:Exercise [Color Extraction]

* Let's find the proper HSV thresholds using an interactive application, and then create a program to extract only the **pink box** from an image.

##### 1. Interactive HSV Checker (`check_hsv.py`)

* Save the following code as `check_hsv.py` and run it. Click around the pink box area in your image to check the average $(H, S, V)$ values in the terminal, and look at the generated gradation map window.

```python
import numpy as np
import cv2

# main function-----------------------------------------------------------------------------
def main():
    global img, cache, bar

    # read image
    img = cv2.imread("./img/static_b.png")
    if img is None:
        print('ERROR: image file is not opened.')
        exit(1)
        
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
    cv2.putText(cimg, "|128", (128, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
    cv2.putText(cimg, "V value", (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
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
        av_b = np.mean(img[max(0, y-2):y+2, max(0, x-2):x+2, 0])
        av_g = np.mean(img[max(0, y-2):y+2, max(0, x-2):x+2, 1])
        av_r = np.mean(img[max(0, y-2):y+2, max(0, x-2):x+2, 2])

        print("(B,G,R) = (" + str(av_b) + ", " + str(av_g) + ", " + str(av_r) + ")")

        # average of HSV components around pointing area
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        av_h = np.mean(hsv[max(0, y-2):y+2, max(0, x-2):x+2, 0])
        av_s = np.mean(hsv[max(0, y-2):y+2, max(0, x-2):x+2, 1])
        av_v = np.mean(hsv[max(0, y-2):y+2, max(0, x-2):x+2, 2])

        print("(H,S,V) = (" + str(av_h) + ", " + str(av_s) + ", " + str(av_v) + ")")
        createGradationImage(av_h, av_s, av_v)

        # track bar
        cv2.namedWindow("color range", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
        if bar == []:
            bar = cv2.createTrackbar("Hue", "color range", int(av_h), 179, changeTrackbarRange)
        else:
            cv2.setTrackbarPos("Hue", "color range", int(av_h))

        # show clicked area
        cv2.rectangle(cache, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), 1)
        cv2.imshow("target image", cache)

if __name__ == '__main__':
    main()

```

##### 2. Extraction Program (`extract_color.py`)

* Based on your measurement results from the interactive tool, define the pink thresholds and apply `cv2.inRange()` to create a binary mask image.

```python
import cv2
import numpy as np

def main():
    # 1. Read the image
    img = cv2.imread("./img/static_b.png")
    if img is None:
        print('ERROR: image file is not opened.')
        exit(1)

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define the range of pink color in HSV
    # [Hue lower-upper, Saturation lower-upper, Value lower-upper]
    lower_pink = np.array([140,  50,  50])
    upper_pink = np.array([170, 255, 255])

    # 4. Create a mask image (pixels within range become 255, others become 0)
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # 5. Extract the pink region using bitwise AND operation
    result = cv2.bitwise_and(img, img, mask=mask)

    # 6. Show the results
    cv2.imshow("Original Image", img)
    cv2.imshow("Mask (White = Pink Area)", mask)
    cv2.imshow("Extracted Pink Box", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

* It's O.K. if the `Extracted Pink Box` window completely separates the pink box from the background space.

---

[back to the top page](../README.md)
