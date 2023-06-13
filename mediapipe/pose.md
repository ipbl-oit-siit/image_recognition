# Tutorials for MediaPipe Pose

[back to the top page](../README.md)

---

## Objectives
This page explains how to make a program for pose detection and get information.

## Prerequisite
You have to finish [Image processing basics for static image](../basics/basics_image.md), [Image processing basics for video](../basics/basics_video.md).

## Pose landmark model
By using [MediaPipe ](https://google.github.io/mediapipe/), we can obtain 3D position information of `33` landmarks as shown by the red marker in the following figure.<br>
<image src="../image/pose_tracking_full_body_landmarks.png" width="75%" height="75%"><br>

## :o:Practice[Display all pose landmarks]
  Get pose landmarks and display them.
  - Execute `py23_ipbl_start.bat` file, and open the VSCode.
  - Click the following button, and make a python file `mypose.py` in the `SourceCode` directory. <br>
  <image src="../image/newfile2_2022.png" width="50%" height="50%"><br>
  - Type the following sample code. It's OK copy and paste.
    
### Sample code
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipePoseLandmark import MediapipePoseLandmark as PoseLmk

device = 0 # cameera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

# Draw a circle on index finger
def my_draw_pose(image, Pose):
    for id_pose in range(Pose.num_detected_poses):
        landmark_point = []
        for i in range(Pose.num_landmark_points):
            landmark_point.append(Pose.get_landmark(id_pose, i))
            cv2.circle(image, (landmark_point[i][0], landmark_point[i][1]), 3, (0, 0, 255), 3)

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

    wname = 'MediaPipe PoseLandmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    Pose = PoseLmk()

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
    
        results = Pose.detect(flipped_frame)

        # Draw the face mesh annotations on the image.
        my_draw_pose(flipped_frame, Pose)
        cv2.imshow(wname, flipped_frame)
 
        # We have some visualize function
        # annotated_frame = Pose.visualize(flipped_frame)
        # annotated_frame = Pose.visualize_with_mp(flipped_frame)
        # cv2.imshow('MediaPipe PoseLandmark', annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()
```
  - Run the sample code with input the following command in the terminal.
```
    C:\\...\SourceCode> python mypose.py
``` 
  <image src="../image/pose.png" width="30%" height="30%"><br>
  - If you want to stop this program, press `q` key while the preview window is active.

### How to refer to landmark information
- The detection results of all poses are stored in the variable `results`, but its data structure is complex. Therefore, we provide several class variables and getter functions in our mediapipe class.
- After `Pose.detect(frame)`, you can call following class variables and getter functions.
  - `Pose.num_detected_poses`: The number of detected poses (max is `2` in the default setting). If `0`, it's dangerous to continue the process. An error may occur when some referencing.
  - `Pose.num_landmark_points`: The number of pose landmarks. Basically, it's `33` in PoseLandmark.
  - `landmark_point = Pose.get_landmark(id_pose, id_landmark)`
    - `landmark_point`: The coordinate array of `id_landmark`-th landmark of `id_pose`-th pose. Type is `np.ndarray([x, y, z], dtype=int)`
  - `presence = Pose.get_landmark_presence(id_pose, id_landmark)`: The presence of `id_landmark`-th landmark of `id_pose`-th pose. If low, the validity is low.
  - `visibility = Pose.get_landmark_visibility(id_pose, id_landmark)`: The presence of `id_landmark`-th landmark of `id_pose`-th pose. If low, the validity is low.
  - sample code
  ```python
  def my_draw_pose(image, Pose):
      for id_pose in range(Pose.num_detected_poses):
          landmark_point = []
          for i in range(Pose.num_landmark_points):
              landmark_point.append(Pose.get_landmark(id_pose, i))
              cv2.circle(image, (landmark_point[i][0], landmark_point[i][1]), 3, (0, 0, 255), 3)
  ```
  
## :o:Exercise[Pose1]
 - Calculate and display the center of gravity from all the obtained landmarks.
 - Define a function that calculates and displays the center of gravity.<br>
    <image src="../image/pose_center.png" width="30%" height="30%"><br>
    - special hint
    ```python
    cog_point = np.zeros((3,), dtype=int)
    for i in range(Pose.num_landmark_points):
        cog_point = cog_point + Pose.get_landmark(id_pose, i)
    cog_point = (cog_point / Pose.num_landmark_points).astype(int)
    ```

## :o:Exercise[Pose2]
 - Calculate and display the center of gravity from all visible landmarks of view.
 - Define a function that calculates and displays them.<br>
    <image src="../image/pose_center.png" width="30%" height="30%"><br>
    - special hint
    ```python
    cog_point = np.zeros((3,), dtype=int)
    cnt = 0
    for i in range(Pose.num_landmark_points):
        if Pose.get_landmark_visibility(id_pose, i) > 0.5:
            cog_point = cog_point + Pose.get_landmark(id_pose, i)
            cnt += 1
    cog_point = (cog_point / cnt).astype(int)
   ```



---

[back to the top page](../README.md)
