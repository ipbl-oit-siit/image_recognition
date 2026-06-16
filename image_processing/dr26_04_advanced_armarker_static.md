# Advanced Image Processing: AR Marker Detection and Overlay

[back to the top page](../README.md)

---
### :red_square: AR Marker (ArUco Marker)

* An **AR marker** (ArUco marker) is a distinct square pattern used in computer vision to determine positions, orientations, and object identities.

#### Features of AR marker
* It is **robust against lighting changes**, making it more reliable than color-based extraction (HSV).
* It allows for **precise coordinate handling**, making it easy to track a physical object's 3D space and orientation.

#### Data Range in OpenCV
When using OpenCV (`cv2.aruco`), the detection system utilizes specific parameters:
* **Dictionary**: Selection of predefined marker pattern sheets (e.g., `DICT_4X4_50`).
* **ID**: The individual identifier assigned to each generated square pattern (e.g., $0$ to $49$).
* **Corners**: A list containing the 4 outer corner coordinates ($x, y$) of the detected marker.

#### :blue_square: Marker detection with `cv2`
* You can detect markers embedded inside an image using the following function:

```python
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, dictionary)

```

#### :o:Practice [AR Marker Detection]

* Save the following sample code as a python file, and execute it. (`C:/oit/home/ipbl/sample_marker.py`)
* `sample_marker.py`

```python

import cv2

def main():
    # 1. Load the scene containing generated markers
    img = cv2.imread("balanced_random_markers.png")
    if img is None:
        print("ERROR: balanced_random_markers.png not found.")
        return

    # 2. Select a 4x4 pixel marker dictionary containing 50 unique IDs
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # 3. Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
    
    # 4. Draw borders around detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        print(f"Detected IDs: {ids.flatten()}")
    
    cv2.imshow('Detected Markers', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

* It is O.K., if the window pops up and you can observe green borders and unique ID labels drawn exactly over the black-and-white markers.

> [!TIP]
>  **Hint: How to measure the "size" (distance) of an AR Marker:**
> 
>  You can estimate how close the marker is by calculating its diagonal distance (pixel distance between Top-Left [0] and Bottom-Right [2]). This value is highly robust against marker rotation and directly corresponds to the physical distance.
> 
> ```Python
> 
> # Calculate diagonal distance using numpy
> size_px = np.linalg.norm(marker_corners[2] - marker_corners[0])
> 
> ```

---

### :red_square: Advanced Application: Image Overlay using AR Marker

By analyzing the four corner coordinates of the detected AR marker, we can calculate a **Perspective Transformation** matrix and warp a target image to fit precisely on top of the marker area.

#### :o:Exercise [AR Marker Overlay]

* Let's complete the program to detect the markers from the scene and overlay matching cat-themed playing card images onto them.

##### 1. Concept of Perspective Overlay

To overlay an image onto the detected marker, we map the four corners of the source card image (`src_pts`) to the four detected marker coordinates (`dst_pts`). By calculating a transformation matrix `M` via `cv2.getPerspectiveTransform()`, we can warp the card image to match the perspective and tilt of the marker in the scene.

> [!NOTE]
>  **Hint for the TODO sections:**
> * `cv2.aruco.detectMarkers` returns a list of corners and an array of detected `ids`. Make sure your output variable names match the validation code (`ids is None`).
> * The `corners` list stores arrays for each detected marker. You need to access the target marker's array index and reshape it to a simple $(4, 2)$ grid.
> 
> 

##### 2. Complete the Overlay Program (`ipB_detectARmarker.py`)

* Open the distributed `ipB_detectARmarker.py` file.
* **Complete the `TODO` sections** to detect markers and apply the perspective warp.

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
        
    # TODO: Detect the markers. Assign output to match the "ids" variable below!
    corners, ids, _ = cv2.aruco.detectMarkers(scene_img, dictionary)

    if ids is None:
        print("No AR markers were detected.")
        return

    # 3. Loop through every detected marker and overlay its designated image
    for i, marker_id in enumerate(ids.flatten()):
        
        # TODO: Extract the 4 corner coordinates for the current marker index "i" and reshape to (4, 2)
        marker_corners = corners[i].reshape((4, 2))
        
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

* It's O.K. if the `Final Card Overlay Result` window completely replaces the black-and-white markers with the warped playing card images seamlessly.

---
[back to the top page](../README.md)

