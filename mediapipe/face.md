# Tutorials for MediaPipe Face

## Objectives
This page explains how to make a program for face detection and get information.

## Prerequisite
You have to finish [MediaPipe Hands](../mediapipe/hands.md).
## Face landmark model
By using [MediaPipe](https://google.github.io/mediapipe/), we can obtain 3D position information of 468 landmarks as shown by the red marker in the following figure.<br>
<image src="../image/face_mesh_android_gpu.gif" width="20%" height="20%">
  <image src="../image/face_landmark2.png" width="38%" height="38%"><br>

## Practice[Display all face landmarks]
  Get face landmarks and display them.
  - Execute "vscode.bat" file, and open the VSCode.
  - Make a python file `myface.py`. 
  - Type the following sample code. It's OK copy and paste.

### Sample code
```python
import cv2
import mediapipe as mp
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

device = 0 # cameera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

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

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    face_mesh = mp_face_mesh.FaceMesh(
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
        results = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
                my_draw_face(frame, face_landmarks)
        cv2.imshow('MediaPipe FaceMesh', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

def my_draw_face(image, landmarks):
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

    if len(landmark_point) != 0:
        for i in range(0, len(landmark_point)):
            cv2.circle(image, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 1)

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\code> python myface.py
``` 
  <image src="../image/face.png" width="30%" height="30%"><br>
  - If you want to stop this program, press "Esc" key while the preview window is active.

### How to refer all the landmarks stored in the list
 - Draw by referring to all the landmarks stored in the list by the following code.
```python
    for i in range(0, len(landmark_point)):
        cv2.circle(image, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 1)
```

## Exercise[Face1]
 - Calculate the center of gravity of all face landmarks, and draw red circle.<br>
    <image src="../image/q1_face.png" width="30%" height="30%"><br>

### :o:Checkpoint
It's OK, you can finish the Exercise[Face1].

---

[README](../README.md)
