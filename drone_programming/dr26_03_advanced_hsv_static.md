# Advanced Image Processing: HSV Color Space

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

* **Hue**: $0$ to $180$ (The actual angle $0$ to $360^\circ$ is divided by 2. For example, Hue = $38$ represents $38 \times 2 = 76^\circ$).
* **Saturation**: $0$ to $255$ ($0$ to $100\%$).
* **Value**: $0$ to $255$ ($0$ to $100\%$).

<div align="center">
  <img src="../image/hsv_circle.png" width="150"><img src="../image/hsv_cylinder.png" width="300">
</div>

#### :blue_square: Color conversion with `cv2`

* You can convert a BGR image to an HSV image using the following function:
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

```



#### :o:Practice[HSV color space]

* Save the following sample code as a python file, and execute it. (`C:/oit/home/ipbl/sample_hsv.py`)
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
- Let's find the proper HSV thresholds of the **pink box** using the interactive tool, and then complete the program to extract it.

##### 1. Find HSV values using `color_picker.py`
- Run the distributed [`color_picker.py`](color_picker.py) program. 
- **Click several different points** inside the pink box (such as the brightest areas, darker shaded areas, and average areas). 
- Observe the $(H, S, V)$ values printed in the terminal each time to find the minimum and maximum values of the pink region.

<div align="center">
  <img src="../image/hsv_clicker.png" width="300">
</div>

##### 2. Concept of Color Extraction
To extract a specific color, we filter the HSV image by defining a lower and upper boundary for each channel. Pixels that fall within this range form a **Binary Mask** (White = Target color, Black = Others). By combining this mask with the original image using a bitwise AND operation, we can isolate the target object.

> [!NOTE]
>  **How to set `lower_pink` and `upper_pink`:**
> 
> Look at the multiple $(H, S, V)$ values you gathered by clicking around the box:
> - **`lower_pink`**: Set values slightly lower than the *minimum* H, S, and V you observed.
> - **`upper_pink`**: Set values slightly higher than the *maximum* H, S, and V you observed.

##### 3. Complete the Extraction Program (`extract_color.py`)
- Save the following code as `extract_color.py`. 
- **Modify `lower_pink` and `upper_pink` values** by inputting the HSV range you discovered in Step 1, then run the program.

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
    # TODO: Input your measured HSV range here! [Hue, Saturation, Value]
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
