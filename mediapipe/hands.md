# Tutorials for MediaPipe Hands

## Objectives
This page explains how to make a program for palm detection and get information.

## Prerequisite
You have to finish [MediaPipe Pose](../mediapipe/pose.md).

## Hand landmark model
By using [MediaPipe](https://google.github.io/mediapipe/), we can obtain 3D position information of 21 landmarks as shown by the red marker in the following figure.<br>
<image src="../image/hand_landmarks.png" width="75%" height="75%"><br>

## Practice[Get information of arbitrary landmark]
  Get information and display about an landmark of index finger.
  - Execute "vscode.bat" file, and open the VSCode.
  - Make a python file `myhands.py` in the `code` directory. <br>
  - Type the following sample code. It's OK copy and paste.

### Sample code
```python
import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = 0 # cameera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

# Draw a circle on index finger
def drawFingertip(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x, y, z to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append(np.array([landmark_x, landmark_y, landmark_z], dtype=int))

    # Draw a circle on index finger and display the coordinate value
    cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), 3)
    cv2.putText(image, "(" + str(landmark_point[8][0]) + ", " + str(landmark_point[8][1]) + ")", 
        (landmark_point[8][0] - 20, landmark_point[8][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


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

    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)

    hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

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
        # the BGR image to RGB.
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = hands.process(frame)

        # Draw the index finger annotation on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawFingertip(frame, hand_landmarks)
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\code> python myhands.py
``` 
  <image src="../image/index_finger_text.png" width="30%" height="30%"><br>
  - If you want to stop this program, press "Esc" key while the preview window is active.
    
### How to refer to landmark information
 - In the above sample code, the detection results of all hands are stored in the variable `results`.
```python
    results = hands.process(image)
```
 - All the obtained landmark information is stored in the variable `results`. 
    You can refer to the landmark information of a specific hand as follows. 
    In the following example, the coordinate value of landmark #4(thumb) of the 0th hand is obtained.
```python
    results.multi_hand_landmarks[0].landmark[4]
```
 - Since the obtained values(`landmark.x`, `landmark.y`) are normalized to [0.0-1.0], it is necessary to convert them to the coordinates of the image as follows.
 - `landmark.z` represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera.
```python
    landmark_x = min(int(landmark.x * image_width), image_width - 1)
    landmark_y = min(int(landmark.y * image_height), image_height - 1)
```
 - In the `drawFingertip` function, the 3D coordinates of each landmark are stored in the list `landmark_point`. 
    The 3D coordinates of the index finger are stored in `landmark_point[8]`, the x-coordinate is stored in `landmark_point[8][0]`, the y-coordinate is stored in `landmark_point[8][1]`, and the z-doordinate is stored in `landmark_point[8][2]`. The number 8 is based on the hand landmark model.
```python
    cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), 3)
    cv2.putText(image, "(" + str(landmark_point[8][0]) + ", " + str(landmark_point[8][1]) + ")", 
        (landmark_point[8][0] - 20, landmark_point[8][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
```
  - `landmark.visibility` indicates whether landmark is visible or occluded by other objects. `landmark.presence` indicates whether landmark is present on the scene.
    Both value ranges are [0.0-1.0].
```python
    if landmark.visibility < 0 or landmark.presence < 0:
        continue
```
    
## Exercise[Hands1]
 - Draw red circles on all fingertips.<br>
    <image src="../image/q1_fingertips.png" width="30%" height="30%"><br>
      
## Exercise[Hands2]
 - Calculate the center of gravity of fingertips, and draw green circle.
 - The center of gravity can be calculated by calculating the average position of the landmark.
 - Since landmarks are sometimes acquired in the wrong position, using the centroid may allow you to create more robust applications.<br>
    <image src="../image/q2_fingertips_c.png" width="30%" height="30%"><br>
### :o:Checkpoint
It's OK, you can finish the Exercise[Hands1] and [Hands2].

## Practice[Calculate the angle between two vectors]
 - To recognize the shape of the finger, it is necessary to calculate the angle between the vectors.
  - Make a python file `calcAngle3D.py`. 
  - Type the following sample code. It's OK copy and paste.
### Sample code
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
    C:\\...\code> python calcAngle3D.py
``` 
 - In this sample code, calculate the angle between vec1 and vec2. The `calcAngle` function takes two vectors as input and returns the angle(degree) between the two vectors.

## Practice[Judgement of whether the finger is bend or not]
 - Judge whether the index finger is open by using the 3D coordinates of the landmark.
  - Make a python file `judgeOpen.py`. 
  - Type the following sample code. It's OK copy and paste.

### Sample code
```python
import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = 0 # cameera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

# check the index finger
def judgeOpen(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x, y, z to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append(np.array([landmark_x, landmark_y, landmark_z], dtype=int))

        if len(landmark_point) != 0 and len(landmark_point)==21:
            vec1 = (landmark_point[5][0] - landmark_point[6][0], 
            landmark_point[5][1] - landmark_point[6][1], 
            landmark_point[5][2] - landmark_point[6][2])
            vec2 = (landmark_point[7][0] - landmark_point[6][0], 
            landmark_point[7][1] - landmark_point[6][1], 
            landmark_point[7][2] - landmark_point[6][2])
            if calcAngle(vec1, vec2) > 140:
                cv2.putText(image, "open", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
            else:
                cv2.putText(image, "bend", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)

            cv2.circle(image, (landmark_point[5][0], landmark_point[5][1]), 7, (0, 0, 255), 3)
            cv2.circle(image, (landmark_point[6][0], landmark_point[6][1]), 7, (0, 0, 255), 3)
            cv2.circle(image, (landmark_point[7][0], landmark_point[7][1]), 7, (0, 0, 255), 3)
            cv2.line(image, (landmark_point[5][0], landmark_point[5][1]), (landmark_point[6][0], landmark_point[6][1]), (0, 255, 0))
            cv2.line(image, (landmark_point[6][0], landmark_point[6][1]), (landmark_point[7][0], landmark_point[7][1]), (0, 255, 0))

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

    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)

    hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7)

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
        # the BGR image to RGB.
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = hands.process(frame)

        # Draw the index finger annotation on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                judgeOpen(frame, hand_landmarks)

        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\code> python judgeOpen.py
``` 
  <image src="../image/open.png" width="30%" height="30%"> <image src="../image/close.png" width="30%" height="30%"><br>
  - If you want to stop this program, press "Esc" key while the preview window is active.                                                           

### How to judge whether your finger is bend or not
 - In this sample code, the angle between the vectors obtained from the joint position of the finger is calculated and judged whether the finger is bend or not.
  - If the angle between vector `landmark_point[5] - landmark_point[6]` and vector `landmark_point[7] - landmark_point[6]` is greater than 140 degrees, display "open".<br>
    <image src="../image/vec.png" width="20%" height="20%"><br>
```python
    if calcAngle(vec1, vec2) > 140:
        cv2.putText(image, "open", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
```

---

[README](../README.md)
