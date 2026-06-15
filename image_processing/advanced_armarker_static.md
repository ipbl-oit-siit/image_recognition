# Advanced Image Processing: AR Marker Generation and Detection

[back to the top page](../README.md)

---
### :orange_square: AR Marker

* An **AR marker** (such as an ArUco marker) is a distinct square pattern used in computer vision to determine positions, orientations, and object identities.



#### Features of AR marker

* It is **reversibly convertible** between black-and-white grid matrices and binary IDs.
* It allows for more **intuitive coordinate handling**, making it easy to specify areas like "a 3D space relative to a physical object".

#### Data Range in OpenCV

When using OpenCV (`cv2`), the system utilizes specific parameters to handle marker generation and object position parameters:

* **Dictionary**: Selection of predefined marker pattern sheets (e.g., `DICT_4X4_50`).
* **ID**: $0$ to $49$ (The individual identifier assigned to each generated square pattern).
* **SidePixels**: Width and height dimensions required for image pixel allocation.



#### :blue_square: Color conversion with `cv2`

* You can detect markers embedded inside a BGR image using the following function:
```python
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, dictionary)

```

#### :o:Practice[AR Marker]

* Save the following sample code as a python file, and execute it. (`C:/oit/py25en/source/sample_marker.py`)
* `sample_marker.py`

```python
import cv2

def main():
    # Select a 4x4 pixel marker dictionary containing 50 unique IDs
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # TODO: Fill appropriate integers into id and sidePixels!
    id = 
    sidePixels = 
    
    # Generate the marker image
    try:
        marker = cv2.aruco.drawMarker(dictionary, id, sidePixels)
    except AttributeError:
        marker = cv2.aruco.generateImageMarker(dictionary, id, sidePixels)
    
    cv2.imwrite('marker.png', marker)

if __name__ == '__main__':
    main()

```

* It is O.K., if the windows pop up and you can observe how the components of Hue, Saturation, and Value are separated.

### :orange_square: Advanced Application: Image Overlay using AR Marker

By analyzing pixel values in the AR marker space, we can define a specific range of coordinates to overlay target objects onto an image.

#### :o:Exercise [AR Marker Overlay]

* Let's find the proper HSV thresholds of the **pink box** using the interactive tool, and then complete the program to extract it.

##### 1. Find HSV values using `color_picker.py`

* Run the distributed [`color_picker.py`](https://www.google.com/search?q=color_picker.py) program.
* **Click several different points** inside the pink box (such as the brightest areas, darker shaded areas, and average areas).
* Observe the $(H, S, V)$ values printed in the terminal each time to find the minimum and maximum values of the pink region.

##### 2. Concept of Color Extraction

To extract a specific color, we filter the HSV image by defining a lower and upper boundary for each channel. Pixels that fall within this range form a **Binary Mask** (White = Target color, Black = Others). By combining this mask with the original image using a bitwise AND operation, we can isolate the target object.

> 💡 **How to set `lower_pink` and `upper_pink`:**
> Look at the multiple $(H, S, V)$ values you gathered by clicking around the box:
> * **`lower_pink`**: Set values slightly lower than the *minimum* H, S, and V you observed.
> * **`upper_pink`**: Set values slightly higher than the *maximum* H, S, and V you observed.
> 
> 

##### 3. Complete the Extraction Program (`extract_color.py`)

* Open the distributed [`ipB_detectARmarker.py`](https://www.google.com/search?q=ipB_detectARmarker.py) file.
* **Complete the `TODO` sections** to detect the markers from the scene image (`balanced_random_markers.png`) and overlay the matching cat-themed playing card images (`0.png` to `6.png`) based on the detected IDs.

```python
import cv2
import numpy as np
import os

def get_card_filename(marker_id):
    """Maps a marker ID to its corresponding playing card image file."""
    if marker_id == 0:
        return '0.png'    # Joker
    elif marker_id == 1:
        return '1cl.png'  # Club Ace
    elif marker_id == 2:
        return '1di.png'  # Diamond Ace
    elif marker_id == 3:
        return '1ht.png'  # Heart Ace
    elif marker_id == 4:
        return '1sp.png'  # Spade Ace
    elif 5 <= marker_id <= 9:
        # Map IDs 5-9 to card images 2.png to 6.png
        return f"{marker_id - 3}.png"
    else:
        return '0.png'

def main():
    # 1. Load the scene containing generated markers
    scene_img = cv2.imread("balanced_random_markers.png")
    if scene_img is None:
        print("ERROR: balanced_random_markers.png not found.")
        return
    
    output_img = scene_img.copy()

    # 2. Initialize the ARuCo detector dictionary
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        dictionary = cv2.aruco.getPredefinedDictionary(0)
        
    # TODO: Detect the markers embedded inside the scene using appropriate variables
    ________, ______, _ = cv2.aruco.detectMarkers(scene_img, dictionary)

    if ids is None:
        print("No AR markers were detected.")
        return

    # 3. Loop through every detected marker and overlay its designated image
    for i, marker_id in enumerate(ids.flatten()):
        
        # TODO: Extract the 4 corner coordinates for the current marker and reshape to (4, 2)
        marker_corners = _________[i].reshape((4, 2))
        
        # Determine the target overlay card file name based on the detected ID
        card_file = get_card_filename(marker_id)
        if not os.path.exists(card_file):
            continue
            
        card_img = cv2.imread(card_file)
        ch, cw = card_img.shape[:2]

        # Define source points from the card image corners
        src_pts = np.array([
            [0, 0],
            [cw - 1, 0],
            [cw - 1, ch - 1],
            [0, ch - 1]
        ], dtype=np.float32)

        # Destination points correspond directly to the detected marker corners
        dst_pts = marker_corners.astype(np.float32)

        # Compute Perspective Transformation matrix and warp the card image
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        sh, sw = scene_img.shape[:2]
        warped_card = cv2.warpPerspective(card_img, M, (sw, sh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        # Create a mask to composite the card onto the scene background
        mask = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
        
        mask_inv = cv2.bitwise_not(mask)
        output_img = cv2.bitwise_and(output_img, output_img, mask=mask_inv)
        card_area = cv2.bitwise_and(warped_card, warped_card, mask=mask)
        output_img = cv2.add(output_img, card_area)

    # 4. Show final result
    cv2.imshow('Final Card Overlay Result', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

* It's O.K. if the `Extracted Pink Box` window completely separates the pink box from the background space.

---

[back to the top page](https://www.google.com/search?q=../README.md)

```
