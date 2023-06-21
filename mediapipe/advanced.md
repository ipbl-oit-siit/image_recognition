# Advanced program

[back to the top page](../README.md)

---

## Objectives
This page contains challenges using all the techniques you have learned.

## Prerequisite
- You have to finish the followings.
    - [Image processing basics for static image](../image_processing/basics_image.md)
    - [Image processing basics for video](../image_processing/basics_video.md)
    - [MediaPipe Pose](../mediapipe/pose.md)
    - [MediaPipe Hand](../mediapipe/hand.md)
    - [MediaPipe Face](../mediapipe/hand.md)

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

## :o:Challenge[Face1]
- Display the face direction randomly, and show how many times you have turned your face to the direction.<br>
 <image src="../image/face_app.gif" width="30%" height="30%"><br>
- Use the following code to randomly generate an integer. In this code, `random.randint(0, 5)` returns a random integer int with `0<=n<=5`.
    ````python
    import random
    random.randint(0, 5)
    ````
- You can randomly select and display the elements of the array with the following code.
    ````python
    msg_array = ("msg1", "msg2", "msg3")
    select = msg_array[random.randint(0, len(msg_array)-1)]
    print(select)
    ````
- Facial orientation can be determined, for example, by the vector from the center of gravity of the all face landmarks to the head of the nose.

---

[back to the top page](../README.md)
