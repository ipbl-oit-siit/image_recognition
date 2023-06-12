# Tutorials for MediaPipe Hands

[back to the top page](../README.md)

---

## Objectives
This page explains how to make a program for palm detection and get information.

## Prerequisite
You have to finish [MediaPipe Pose](../mediapipe/pose.md).

## Hand landmark model
By using [MediaPipe](https://google.github.io/mediapipe/), we can obtain 3D position information of 21 landmarks as shown by the red marker in the following figure.<br>
<image src="../image/hand_landmarks.png" width="75%" height="75%"><br>

## :o:Practice[Get information of arbitrary landmark]
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

# Draw a circle on index finger
def my_draw_fingertip(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        landmark_point = []
        for i in range(Hand.num_landmark_points):
            landmark_point.append(Hand.get_landmark(id_hand, i))

        # Draw a circle on index finger and display the coordinate value
        id_lmk = Hand.INDEX_FINGER_TIP # INDEX_FINGER_TIP = 8
        if Hand.num_landmark_points > id_lmk:
            cv2.circle(image, (landmark_point[id_lmk][0], landmark_point[id_lmk][1]), 7, (0, 0, 255), 3)
            txt = '({:d}, {:d})'.format(landmark_point[id_lmk][0], landmark_point[id_lmk][1])
            wrist_point_for_text = (landmark_point[id_lmk][0]-20, landmark_point[id_lmk][1]-20)
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

        # Flip the image horizontally for a later selfie-view display, and convert
        flipped_frame = cv2.flip(frame, 1)
    
        results = Hand.detect(flipped_frame)

        # Draw the face mesh annotations on the image.
        for i in range(Hand.num_detected_hands):
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
  <image src="../image/index_finger_text.png" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.
    
### How to refer to landmark information
- The detection results of all hands are stored in the variable `results`, but its data structure is complex. Therefore, we provide several class variables and getter functions in our mediapipe class.
- After `Hand.detect(flipped_frame)`, you can call following class variables and getter functions.
  - `Hand.num_detected_hands`: The number of detected hands (max is `2` in the default setting). If `0`, it's dangerous to continue the process. An error may occur when some referencing.
  - `Hand.num_landmark_points`: The number of hand landmarks. Basically, it's `21` in HandLandmark.
  - `landmark_point = Hand.get_landmark(id_hand, id_landmark)`
    - `landmark_point`: The coordinate array of `id_landmark`-th landmark of `id_hand`-th hand. Type is `np.ndarray([x, y, z], dtype=int)`
  - `presence = Hand.get_landmark_presence(id_hand, id_landmark)`: The presence of `id_landmark`-th landmark of `id_hand`-th hand. If low, the validity is low.
  - `visibility = Hand.get_landmark_visibility(id_hand, id_landmark)`: The presence of `id_landmark`-th landmark of `id_hand`-th hand. If low, the validity is low.
 - In the `drawFingertip` function, the 3D coordinates of each landmark are stored in the list `landmark_point`. 
    The 3D coordinates of the index finger's TIP are stored in `landmark_point[8]`, the x-coordinate is stored in `landmark_point[8][0]`, the y-coordinate is stored in `landmark_point[8][1]`, and the z-doordinate is stored in `landmark_point[8][2]`. (`INDEX_FINGER_TIP = 8`)
```python
id_lmk = Hand.INDEX_FINGER_TIP # INDEX_FINGER_TIP = 8
if Hand.num_landmark_points > id_lmk:
    cv2.circle(image, (landmark_point[id_lmk][0], landmark_point[id_lmk][1]), 7, (0, 0, 255), 3)
    txt = '({:d}, {:d})'.format(landmark_point[id_lmk][0], landmark_point[id_lmk][1])
    wrist_point_for_text = (landmark_point[id_lmk][0]-20, landmark_point[id_lmk][1]-20)
    cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
```

## :o:Exercise[Hands1]
- Draw red circles on all fingertips. (red: `(0, 0, 255)`)<br>
    <image src="../image/q1_fingertips.png" width="30%" height="30%"><br>
    - hint
    ```python
    for id_lmk in [4, 8, 12, 16, 20]:
    ```

## :o:Exercise[Hands2]
 - Calculate the center of gravity of fingertips, and draw green circle.
 - The center of gravity can be calculated by calculating the average position of the landmark.
 - Since landmarks are sometimes acquired in the wrong position, using the centroid may allow you to create more robust applications.<br>
    <image src="../image/q2_fingertips_c.png" width="30%" height="30%"><br>
    - special hint
    ```python
    cog_point = np.zeros((3,), dtype=int)
    for i in [4, 8, 12, 16, 20]:
        cog_point = cog_point + Hand.get_landmark(id_hand, i)
    cog_point = (cog_point / 5).astype(int)
    ```

## :o:Practice[Calculate the angle between two vectors]
 - To recognize the shape of the finger, it is necessary to calculate the angle between the vectors.
  - Make a python file `calcAngle3D.py`. 
  - Type the following sample code. It's OK copy and paste.
### `calcAngle3D.py`
```python
import numpy as np

def calcAngle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))

def main():
    vec1 = np.array([1, 1, 1])
    vec2 = np.array([1, 1, 0])
    print(calcAngle(vec1, vec2))  

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\SourceCode> python calcAngle3D.py
``` 
 - In this sample code, calculate the angle between vec1 and vec2. The `calcAngle` function takes two vectors as input and returns the angle(degree) between the two vectors.

## :o:Practice[Judgement of whether the finger is bend or not]
 - Judge whether the index finger is open by using the 3D coordinates of the landmark.
  - Make a python file `hands_judge_open.py`. 
  - Type the following sample code. It's OK copy and paste.

### `hands_judge_open.py`
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk


device = 0 # cameera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def judgeOpen(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        landmark_point = []
        for id_lmk in range(Hand.num_landmark_points):
            landmark_point.append(Hand.get_landmark(id_hand, id_lmk))

        if len(landmark_point) == 21: # 21 is the max number of hand landmarks
            
            vec1 = (
                landmark_point[Hand.INDEX_FINGER_MCP][0] - landmark_point[Hand.INDEX_FINGER_PIP][0], 
                landmark_point[Hand.INDEX_FINGER_MCP][1] - landmark_point[Hand.INDEX_FINGER_PIP][1], 
                landmark_point[Hand.INDEX_FINGER_MCP][2] - landmark_point[Hand.INDEX_FINGER_PIP][2]
                )
            vec2 = (
                landmark_point[Hand.INDEX_FINGER_DIP][0] - landmark_point[Hand.INDEX_FINGER_PIP][0], 
                landmark_point[Hand.INDEX_FINGER_DIP][1] - landmark_point[Hand.INDEX_FINGER_PIP][1], 
                landmark_point[Hand.INDEX_FINGER_DIP][2] - landmark_point[Hand.INDEX_FINGER_PIP][2]
                )
            point_for_text = (landmark_point[Hand.INDEX_FINGER_MCP][0]+10, landmark_point[Hand.INDEX_FINGER_MCP][1])
            if calcAngle(vec1, vec2) > 140:
                cv2.putText(image, "open", point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(image, "bend", point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.circle(image, tuple(landmark_point[Hand.INDEX_FINGER_MCP][:2]), 7, (0, 0, 255), 3)
            cv2.circle(image, tuple(landmark_point[Hand.INDEX_FINGER_PIP][:2]), 7, (0, 0, 255), 3)
            cv2.circle(image, tuple(landmark_point[Hand.INDEX_FINGER_DIP][:2]), 7, (0, 0, 255), 3)
            cv2.line(image, tuple(landmark_point[Hand.INDEX_FINGER_MCP][:2]), tuple(landmark_point[Hand.INDEX_FINGER_PIP][:2]), (0, 255, 0))
            cv2.line(image, tuple(landmark_point[Hand.INDEX_FINGER_PIP][:2]), tuple(landmark_point[Hand.INDEX_FINGER_DIP][:2]), (0, 255, 0))

def calcAngle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)
    return np.rad2deg(np.arccos(cos_theta))


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
        frame_now=getFrameNumber(start, fps)
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

        # Draw the face mesh annotations on the image.
        judgeOpen(flipped_frame, Hand)
        cv2.imshow('MediaPipe HandLandmark', flipped_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\SourceCode> python judgeOpen.py
``` 
  <image src="../image/open.png" width="30%" height="30%"> <image src="../image/close.png" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.                                                           

### How to judge whether your finger is bend or not
 - In this sample code, the angle between the vectors obtained from the joint position of the finger is calculated and judged whether the finger is bend or not.
  - If the angle between vector `landmark_point[5] - landmark_point[6]` and vector `landmark_point[7] - landmark_point[6]` is greater than 140 degrees, display "open".
    - `INDEX_FINGER_MCP = 5`, `INDEX_FINGER_PIP = 6`, `INDEX_FINGER_DIP = 7`.<br>
    <image src="../image/vec.png" width="20%" height="20%"><br>
```python
point_for_text = (landmark_point[Hand.INDEX_FINGER_MCP][0]+10, landmark_point[Hand.INDEX_FINGER_MCP][1])
if calcAngle(vec1, vec2) > 140:
    cv2.putText(image, "open", point_for_text, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
```

## :o:Practice[MediapipeHandGestureRecognize Class]
 - To recognize the shape of the hand, you can use `MediapipeHandGestureRecognize Class`.
  - Make a python file `showGestureName.py`. 
  - Type the following sample code. It's OK copy and paste.
### `showGestureName.py`
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

        # Flip the image horizontally for a later selfie-view display, and convert
        flipped_frame = cv2.flip(frame, 1)
    
        results = Hand.detect(flipped_frame)

        # Draw the face mesh annotations on the image.
        my_draw_gesture(flipped_frame, Hand)
        cv2.imshow(wname, flipped_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\SourceCode> python showGestureName.py
``` 
### How to get the hand gesture name
 - `MediapipeHandGestureRecognize Class` inharits `MediapipeHandLandmark Class`, and has some additional getter function like `get_gesture`. 
```python
txt = Hand.get_gesture(id_hand)
```


---

[back to the top page](../README.md)
