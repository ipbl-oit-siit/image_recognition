# Tutorials for MediaPipe Face

[back to the top page](../README.md)

---

## Objectives
This page explains how to make a program for face detection and get information.

## Prerequisite
You have to finish [MediaPipe Hands](../mediapipe/hands.md).

## ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) Face landmark model
By using [MediaPipe Face Landmark](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#get_started), we can obtain 3D position information of 468 landmarks as shown by the red marker in the following figure.<br>
<image src="../image/face_mesh_android_gpu.gif" width="20%" height="20%">
  <image src="https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png" width="28%" height="28%"><br>

## :o:Practice[Display all face landmarks]
  Get face landmarks and display them.
  - Execute `py23_ibpl_start.bat` file, and open the VSCode.
  - Make a python file `myfacelmk.py`. 
  - Type the following sample code. It's OK copy and paste.

### `myfacelmk.py`
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeFaceLandmark import MediapipeFaceLandmark as FaceLmk

device = 0 # cameera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

def my_draw_face(image, Face, id_face):
    landmark_point = []
    for i in range(Face.num_landmark_points):
        landmark_point.append(Face.get_landmark(id_face, i))

    for i in range(len(landmark_point)):
        # Convert the obtained landmark values x, y, z to the coordinates on the image
        cv2.circle(image, (int(landmark_point[i][0]), int(landmark_point[i][1])), 1, (0, 255, 0), 1)

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

    Face = FaceLmk()

    while cap.isOpened():
        frame_now = getFrameNumber(start, fps)
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

        Face.detect(flipped_frame)

        # Draw the face mesh annotations on the image.
        for i in range(Face.num_detected_faces):
                my_draw_face(flipped_frame, Face, i)
        cv2.imshow('MediaPipe FaceLandmark', flipped_frame)
 
        # We have some visualize function (visualize all face landmarks)
        # annotated_frame = Face.visualize(flipped_frame)
        # annotated_frame = Face.visualize_with_mp(flipped_frame)
        # cv2.imshow('MediaPipe FaceLandmark', annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\SourceCode> python myfacelmk.py
``` 
  <image src="../image/face.png" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.

### How to refer all the landmarks stored in the list
- Draw by referring to all the landmarks of the 'id_face'-th face by the following code.
```python
    for i in range(len(landmark_point)):
        cv2.circle(image, (int(landmark_point[i][0]), int(landmark_point[i][1])), 1, (0, 255, 0), 1)
```

## :o:Exercise[Face1]
- Calculate the center of gravity of all face landmarks, and draw red circle.<br>
    <image src="../image/q1_face.png" width="30%" height="30%"><br>
### sample code
```python
def my_draw_cog_point(image, Face, id_face):
    landmark_point = []
    cog_point = np.zeros((3,), dtype=int)
    for i in range(Face.num_landmark_points):
        landmark_point.append(Face.get_landmark(id_face, i))
        cog_point = cog_point + landmark_point[i]
    # calculate the center of gravity point and cast type (float -> int)
    cog_point = (cog_point / Face.num_landmark_points).astype(int)
    cv2.circle(image, (cog_point[0], cog_point[1]), 5, (0, 0, 255), 2)

    txt = '({:d}, {:d})'.format(cog_point[0], cog_point[1])
    wrist_point_for_text = (cog_point[0]-20, cog_point[1]-20)
    cv2.putText(image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
```

---

[back to the top page](../README.md)
