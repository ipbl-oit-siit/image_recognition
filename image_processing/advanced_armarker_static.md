### Advanced Application: AR Marker Generation and Detection

An **AR marker** (such as an ArUco marker) is a distinct square pattern used in computer vision to determine positions, orientations, and object identities.

#### :o:Exercise [AR Marker Overlay]
* Let's understand how to generate and detect AR markers using OpenCV, and then complete a program to overlay dynamic card images on top of detected markers.

##### 1. Generating an AR Marker (`ipB_generateARmarker.py`)
* To create a marker, you select a predefined dictionary (a set of marker patterns) and specify a unique marker ID along with the output pixel size.
* Save the following code as `ipB_generateARmarker.py`.
* **Complete the `TODO` sections** by filling integers into `id` and `sidePixels` to generate your marker image.

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

---

##### 2. Concept of AR Marker Detection

* When capturing or reading an image containing markers, the system returns their exact corner locations and identified IDs.

```python
# Detect markers in an image
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# TODO: Fill variables to catch the return values!
________, ______, rejectedImgPoints = cv2.aruco.detectMarkers(img, dictionary)

# Draw green borders and ID text over the detected markers (Overwrites 'img')
cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))

```

> 💡 **Key Variables to Use:**
> Out of the returned data, we focus only on **`corners`** and **`ids`**:
> * **`corners`**: The coordinate pairs of the detected marker's four corners.
> * **`ids`**: The unique marker ID integer mapped to each detected pattern.
> 
> 

> 🔑 **Supplementary Note: Multiple Return Values in Python**
> The function `cv2.aruco.detectMarkers()` returns multiple outputs simultaneously. Python handles multiple return values smoothly by unpacking them directly into variables. You have already encountered this feature across other OpenCV operations:
> * **Splitting Color Channels**: `b, g, r = cv2.split(img)`
> * **Reading Video Frames**: `ret, frame = cap.read()`
> * **Getting Image Dimensions**: `h, w, c = img.shape`
> 
> 

---

##### 3. Complete the Overlay Program (`ipB_detectARmarker.py`)

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

* It's O.K. if your final display window cleanly places the cute matching card items exactly over each scrambled black-and-white square target area!

```

```
