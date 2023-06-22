# Samples for MediaPipe Hands

[back to the top page](../README.md)

---
## Prerequisite
- You have to finish [Image processing basics for static image](../image_processing/basics_image.md), [Image processing basics for video](../image_processing/basics_video.md).
- Additionally, it is recommended that [MediaPipe Pose](../mediapipe/pose.md) is completed.

## ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) Hand landmark model
By using [MediaPipe Hand](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker), we can obtain 3D position information of `21` landmarks as shown by the red marker in the following figure.<br>
<image src="../image/hand_landmarks.png" width="75%" height="75%"><br>

## :o:Sample[Get information of arbitrary landmark]
Get information and display about an landmark of index finger.
  - Execute `py23_ipbl_start.bat` file, and open the VSCode.
  - Make a python file `myhands.py` in the `SourceCode` directory. <br>
  - Type the following sample code. It's OK copy and paste.

### `myhands.py`
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

# Draw a circle on all finger
def my_draw_fingers(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        for id_lmk in range(Hand.num_landmarks):
            landmark_point = Hand.get_landmark(id_hand, id_lmk)
            cv2.circle(image, (landmark_point[0], landmark_point[1]), 7, (0, 255, 0), 3)

# Draw a circle on index finger
def my_draw_fingertip(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        # Draw a circle on index finger and display the coordinate value
        id_lmk = Hand.INDEX_FINGER_TIP # INDEX_FINGER_TIP = 8
        landmark_point = Hand.get_landmark(id_hand, id_lmk)
        cv2.circle(image, (landmark_point[0], landmark_point[1]), 7, (0, 0, 255), 3)
        # Write text on image
        txt = '({:d}, {:d})'.format(landmark_point[0], landmark_point[1])
        wrist_point_for_text = (landmark_point[0]-20, landmark_point[1]-20)
        cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    cv2.namedWindow('MediaPipe HandLandmark', cv2.WINDOW_NORMAL)

    Hand = HandLmk()

    while cap.isOpened():
        frame_now = get_frame_number(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally
        flipped_frame = cv2.flip(frame, 1) ### very important ####

        results = Hand.detect(flipped_frame)

        # Draw the annotations on the image.
        for i in range(Hand.num_detected_hands):
            my_draw_fingers(flipped_frame, Hand)
            my_draw_fingertip(flipped_frame, Hand)
        cv2.imshow('MediaPipe HandLandmark', flipped_frame)

        # We have some visualize function
        # annotated_frame = Hand.visualize(flipped_frame)
        # annotated_frame = Hand.visualize_with_mp(flipped_frame)
        # cv2.imshow('MediaPipe HandLandmark', annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
    ```
        C:\\...\SourceCode> python myhands.py
    ```
    <image src="../image/myhand.jpg" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.

### How to refer to landmark information
- The detection results of all hands are stored in the variable `results`, but its data structure is complex. Therefore, we provide several class variables and getter functions in our mediapipe class.
- After `Hand.detect(flipped_frame)`, you can call following class variables and getter functions.
  - `Hand.num_detected_hands`: The number of detected hands (max is `2` in the default setting). If `0`, it's dangerous to continue the process. An error may occur when some referencing.
  - `Hand.num_landmarks`: The number of hand landmarks. Basically, it's `21` in HandLandmark.
  - `landmark_point = Hand.get_landmark(id_hand, id_landmark)`
    - `landmark_point`: The coordinate array of `id_landmark`-th landmark of `id_hand`-th hand. Type is `np.ndarray([x, y, z], dtype=int)`
  - `presence = Hand.get_landmark_presence(id_hand, id_landmark)`: The presence of `id_landmark`-th landmark of `id_hand`-th hand. If low, the validity is low.
  - `visibility = Hand.get_landmark_visibility(id_hand, id_landmark)`: The presence of `id_landmark`-th landmark of `id_hand`-th hand. If low, the validity is low.
 - In the `drawFingertip` function, the 3D coordinates of each landmark are stored in the list `landmark_point`.
    The 3D coordinates of the index finger's TIP are stored in `landmark_point[8]`, the x-coordinate is stored in `landmark_point[8][0]`, the y-coordinate is stored in `landmark_point[8][1]`, and the z-doordinate is stored in `landmark_point[8][2]`. (`INDEX_FINGER_TIP = 8`)
    ```python
    id_lmk = Hand.INDEX_FINGER_TIP # INDEX_FINGER_TIP = 8
    landmark_point = Hand.get_landmark(id_hand, id_lmk)
    cv2.circle(image, (landmark_point[0], landmark_point[1]), 7, (0, 0, 255), 3)
    # Write text on image
    txt = '({:d}, {:d})'.format(landmark_point[0], landmark_point[1])
    wrist_point_for_text = (landmark_point[0]-20, landmark_point[1]-20)
    cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
    ```

## :o:Exercise[Hands1]
- Draw red circles on all fingertips. (red: `(0, 0, 255)`)<br>
    <image src="../image/myhand_Hand1.jpg" width="30%" height="30%"><br>
    - hint
    ```python
    for id_lmk in [4, 8, 12, 16, 20]:
        landmark_point = Hand.get_landmark(id_hand, id_lmk)
    ```

## :o:Exercise[Hands2]
 - Calculate the center of gravity of fingertips, and draw green circle. (green: `(0, 255, 0)`)
 - The center of gravity can be calculated by calculating the average position of the landmark.
 - Since landmarks are sometimes acquired in the wrong position, using the centroid may allow you to create more robust applications.<br>
    <image src="../image/myhand_Hand2.jpg" width="30%" height="30%"><br>
    - special hint
    ```python
    cog_point = np.zeros((3,), dtype=int)
    for id_lmk in [4, 8, 12, 16, 20]:
        cog_point = cog_point + Hand.get_landmark(id_hand, id_lmk)
    cog_point = (cog_point / 5).astype(int)
    ```

## :o:Practice[Calculate the angle between two vectors]
 - To recognize the shape of the finger, it is necessary to calculate the angle between the vectors.
  - Make a python file `calc_angle.py`.
  - Type the following sample code. It's OK copy and paste.
### `calc_angle.py`
```python
import numpy as np

def calc_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))

def main():
    vec1 = np.array([0, 0, 1]) # vec1 on z-axis
    vec2 = np.array([0, 1, 0]) # vec2 on y-axis
    print(calc_angle(vec1, vec2))

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
    ```sh
        C:\\...\SourceCode> python calc_angle.py
    ```
 - In this sample code, calculate the angle between vec1 and vec2. The `calc_angle` function takes two vectors as input and returns the angle(degree) between the two vectors.

## :o:Practice[Hand3: Judgement of whether the finger is bend or not]
 - Judge whether the index finger is open by using the 3D coordinates of the landmark.
  - Make a python file `myhand_judge_open.py`.
  - Type the following sample code. It's OK copy and paste.

### `myhand_judge_open.py`
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def calc_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))

def judge_open(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        # pickup landmark points of index finger
        ifmcp_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_MCP)
        ifpip_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP)
        ifdip_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_DIP)

        # draw index finger (MCP - PIP - DIP)
        cv2.circle(image, ifmcp_landmark_point[:2], 7, (0, 0, 255), 3)
        cv2.circle(image, ifpip_landmark_point[:2], 7, (0, 0, 255), 3)
        cv2.circle(image, ifdip_landmark_point[:2], 7, (0, 0, 255), 3)
        cv2.line(image, ifmcp_landmark_point[:2], ifpip_landmark_point[:2], (0, 255, 0))
        cv2.line(image, ifpip_landmark_point[:2], ifdip_landmark_point[:2], (0, 255, 0))

        # make vector
        vec1 = ifmcp_landmark_point - ifpip_landmark_point
        vec2 = ifdip_landmark_point - ifpip_landmark_point
        # calc angle and judge open/bend
        point_for_text = (ifmcp_landmark_point[0]+10, ifmcp_landmark_point[1])
        if calc_angle(vec1, vec2) > 140:
            cv2.putText(image, "open", point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(image, "bend", point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    wname = 'MediaPipe HandLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    Hand = HandLmk()

    while cap.isOpened():
        frame_now=get_frame_number(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally
        flipped_frame = cv2.flip(frame, 1)

        results = Hand.detect(flipped_frame)

        # Draw the annotations on the image.
        judge_open(flipped_frame, Hand)
        cv2.imshow(wname, flipped_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
    ```sh
    C:\\...\SourceCode> python myhand_judge_open.py
    ```
  <image src="../image/myhand_Hand3.jpg" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.

### How to judge whether your finger is bend or not
 - In this sample code, the angle between the vectors obtained from the joint position of the finger is calculated and judged whether the finger is bend or not.
  - Get landmark points of index finger (MCP - PIP - DIP)
    ```python
    ifmcp_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_MCP) # 5
    ifpip_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_PIP) # 6
    ifdip_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_DIP) # 7
    ```
    <image src="../image/vec.png" width="20%" height="20%"><br>
    - `INDEX_FINGER_MCP = 5`, `INDEX_FINGER_PIP = 6`, `INDEX_FINGER_DIP = 7`.<br>
  - If the angle between vector `ifmcp_landmark_point - ifpip_landmark_point` and vector `ifdip_landmark_point - ifpip_landmark_point` is greater than 140 degrees, display "open".
    ```python
    point_for_text = (ifmcp_landmark_point[0]+10, ifmcp_landmark_point[1])
    if calc_angle(vec1, vec2) > 140:
        cv2.putText(image, "open", point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    ```

## :o:Practice[Hand4: MediapipeHandGestureRecognize Class]
 - To recognize the shape of the hand, you can use `MediapipeHandGestureRecognize Class`.
  - Make a python file `myhand_show_gesture_name.py`.
  - Type the following sample code. It's OK copy and paste.
### `myhand_show_gesture_name.py`
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandGestureRecognition import MediapipeHandGestureRecognition as HandGes

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def my_draw_gesture(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        txt = Hand.get_gesture(id_hand)
        wrist_point = Hand.get_landmark(id_hand, Hand.WRIST)
        point_for_text = (wrist_point[0]+10, wrist_point[1]+30)
        cv2.putText(image, org=point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    wname = 'MediaPipe HandGesture'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    Hand = HandGes()

    while cap.isOpened():
        frame_now = get_frame_number(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally
        flipped_frame = cv2.flip(frame, 1)

        results = Hand.detect(flipped_frame)

        # Draw the annotations on the image.
        my_draw_gesture(flipped_frame, Hand)
        cv2.imshow(wname, flipped_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
    ```sh
    C:\\...\SourceCode> python myhand_show_gesture_name.py
    ```
  <image src="../image/myhand_Hand4.jpg" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.

### How to get the hand gesture name
 - `MediapipeHandGestureRecognize Class` inharits `MediapipeHandLandmark Class`, and has some additional getter function like `get_gesture`.
    ```python
    txt = Hand.get_gesture(id_hand)
    ```

## :o:Challenge[Hands1]
 - Create an interactive simple game using the information from **hands** by referring to the sample code below.
### `myhands_simplegame.py`
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk
import random

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def calc_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))

# check the angle between the vertical upward direction and the direction pointed by the index finger
def check_angle(image, Hand, id_hand):
    angle = 0

    ifmcp_landmark_point = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_MCP)

    point_for_text = (ifmcp_landmark_point[0]+10, ifmcp_landmark_point[1])
    cv2.putText(image, str(int(angle)), point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return angle

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    wname = 'MediaPipe HandLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    Hand = HandLmk()

    flag = 0
    quest = random.randint(1, 359)
    msg1 = ""
    msg2 = ""
    while cap.isOpened():
        frame_now = get_frame_number(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        flipped_frame = cv2.flip(frame, 1)

        results = Hand.detect(flipped_frame)

        # Display the message
        cv2.rectangle(flipped_frame, (0, 80), (int(wt), 110), (255,255,255), -1)
        msg1 = "Point in the direction of " + str(quest) + " degrees"
        cv2.putText(flipped_frame, "Reset[r key], Exit[q key]", (100, int(ht)-50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),1)
        cv2.putText(flipped_frame, msg1, (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),1)
        cv2.putText(flipped_frame, msg2, (100, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0,255,0),3)
        if Hand.num_detected_hands > 0:
            id_hand = 0
            if flag == 0:
                if int(checkAngle(flipped_frame, Hand, id_hand)) == quest:
                    flag = 1

        cv2.imshow(wname, flipped_frame)

        if flag == 1:
            msg2 = "ok!"
        key = cv2.waitKey(5) & 0xFF
        if key == ord('r'):
            flag = 0
            msg2 = ""
            quest = random.randint(1, 359)
        elif key == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
```
- In this sample code, 1 to 359 degrees are displayed randomly, and "OK!" is displayed when the user points to the same angle.
- In the `check_angle` function, the angle between the vertical upward direction `vec = [0, -1]` and the index finger `iftip <- ifdip` is calculated.
    - hint
        ```python
        # [180 <- 0 -> 180] --> [0 <-> 360]
        if(iftip_landmark_point[0] - ifdip_landmark_point[0] < 0):
            angle = 360 - angle
        ```
 - **Complete the `check_angle` function.**<br>
  <image src="../image/angle_app.gif" width="30%" height="30%"><br>

---

[back to the top page](../README.md)
