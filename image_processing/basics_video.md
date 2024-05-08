# Image processing basics for video image

[back to the top page](../README.md)

---

## Objectives
- This page explains how to process the video image in Python 3 with your camera device.
- This page explains how to detect the face/facial landmarks with OpenCV.
- This page explains how to process the face/facial landmarks detection on the video image.

## Prerequisite
- Open the VSCode by the running the `ipbl24_start` on the Desktop. Confirm that the current directory shown in the terminal window is `ipbl24`.
- The python program (.py) has to be made in `ipbl24` folder. And all image files are saved (downloaded) in `img` folder and read from there.
- You can run a python program with the input of the following command in the terminal.
    ```sh
    C:\\...\ipbl24> python XXX.py
    ```
## :red_square: Sample of simple video-image processing

### video_viewer1.py
```python
import os
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

device = 0 # camera device number

# main----------------------------------------------------
def main():
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)
    while cap.isOpened() :
        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()
```

### Video properties
- A value of grobal variable `device` in `video_viewer1.py` is the device number of the camera starting from 0.
    - It can be read the movie file when a value of the `device` is set a movie file name, like the following.
        ```python
        device = "./img/moviefile.avi"
        ```
- The following codes in `video_viewer1.py` are in order to open the video stream and get the properties of the video-image.
    | code | comment |
    :--- | :---
    | cv2.VideoCapture(device) | Open the video stream |
    | cv2.CAP_PROP_FPS | FPS (frame / sec) rate |
    | cv2.CAP_PROP_FRAME_WIDTH | Frame width |
    | cv2.CAP_PROP_FRAME_HEIGHT | Frame height |

### Read a frame from video stream
- The following function in `video_viewer1.py` is in order to read a frame form video steram.
    | code | comment |
    :--- | :---
    | cap.read() | 1st return value is a boolean value for whether success in reaing a frame <br> 2nd return value is the list of the pixel values in a frame |
    - `cap.read()` function is called by every loop in `video_viewer1.py`, independent of the FPS.
    - The time of the loop is costed the sum of the processing time with `cap.read()` and any other functions in the while block and the sleep time with `cv2.waitKey()` function (1m sec).
        ```python
        while cap.isOpened() :
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow("video", frame)
        ```
- In the video stream, the returned frame from `cap.read()` is the same as the previous frame when the time of the `cap.read()` function calling is shorter than 1/FPS sec.
- In the movie file, the returned frame from `cap.read()` function is in order, independent on the time of function calling.
    | Device | The number of the returned frame |
    :--- | :---
    | camera | `t + int( time of the "read" function calling / fps)` |
    | movie file | `t + 1` |

### Wait for the user's key input
- `cv2.waitKey()` function sleeps the process(thread) in order to wait for the user's key input during a value of the argument (m sec).
- It exits the while loop when the user presses the `q` key.

### :o:Practice
- You should be copy [`video_viewer1.py`](#video_viewer1py) with the `clipboard` button and paste it to the VS Code, and save it as  `video_viewer1.py` in the `ipbl24` folder.
- If there is only one camera on your device, including the built-in, the number of your camera device is `0`.
- Set a value of the global variable `device` to adapt your PC environment.
- Run the sample code.
    - It can be ignored if a warning message like the following will appear.
         ```
         [ WARN:0] global C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-sxpsnzt6\opencv\modules\videoio\src\cap_msmf.cpp (435) `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback
         ```
- Check the video window is came up, and the program is terminated with the `q` key press.



## :red_square: Sample of video-image processing adapted the frame rate

### Add the function to calculate the frame number from the processing time
- Define the function for the frame number calculation in order to avoid to be called `cap.read()` function multiple times, independent on the FPS.
- The processes after `cap.read()` function in the while block are skipped and continued the iteration if the next iteration is came up before the next frame is provided from the video stream.

### video_viewer2.py
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time

device = 0 # camera device number

# added function -----------------------------------------
def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)
    return frame_now

# main----------------------------------------------------
def main():
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1
    while cap.isOpened() :
        frame_now=getFrameNumber(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()
```

### :o:Checkpoint (Sample of video-image processing adapted the frame rate)
- It's OK, you confirm only to be able to run the program.
- Which program, [`video_viewer2.py`](#video_viewer2py) or [`video_viewer1.py`](#video_viewer1py), is better to use is dependent on the situation.
- There is a more simple way for adapting the frame rate that a value of the argument in `cv2.waitKey()` function in the [`video_viewer1.py`](#video_viewer1py) is replaced to the time until the next frame is provided, like the following.
  ```python
  if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
  ```

### :o:Exercise (`selfie.py`)
- Try to make "Let's selfie program" (`selfie.py`) by modifying the [`video_viewer1.py`](#video_viewer1py) or the [`video_viewer2.py`](#video_viewer2py).
- Save a video frame to a still image file at the time when the user presses `s` key.
    | Key | Details |
    :---: | :---
    | q | The program is terminated. |
    | s | The video frame is saved as a still image. |

- Here is the hint code.
  ```python
  key = cv2.waitKey(1)
  if key & 0xFF == ord('q'):
      break
  elif key & 0xFF == ord('s'):
      cv2.imshow("video", frame)
      cv2.imwrite("./img/selfie.jpg", frame)
  ```
  - The following function in the hint code is in order to write a still image.

    | code | comment |
    :--- | :---
    | `cv2.imwrite("name", variable)` | 1st argument is the file name(path) of the image which is saved. <br>2nd argument is the variable of the image. |

  - The following function in the hint code is in order to compare a received key with a key that you want to detect.

    | code | comment |
    :--- | :---
    | `ord('a caracter')` | It's changed a character in the argument to the number of Unicode. |

- If your program is correct, you will be able to find a jpeg file named `selfie.jpg` in `img` folder when you press `s` key.

## :red_square: Appendix: Sample of video-image recorder
### Preparation
- Download [`openh264-1.8.0-win64.dll`](https://oskit-my.sharepoint.com/:u:/g/personal/yoshiyuki_kamakura_oit_ac_jp/EbP0DohYsAtHt9sXCA22EZ8BR1LDrgoMHTeAv_ihERy_Kg?e=HVQlFv), and save it in `ipbl24` folder.
    - It's a H264 codec file for Windows.

### video_recorder.py
```python
# Sample of video-image recorder
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

device = 0 # camera device number
video_name = "record.avi"

# main----------------------------------------------------
def main():
    global device, video_name
    recflag = False

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"H264"), fps, (int(wt), int(ht)))

    while cap.isOpened() :
        ret, frame = cap.read()
        if recflag:
            writer.write(frame)
            cv2.circle(frame, (20, 20), 5, [0,0,255], -1)


        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            recflag = ~recflag # `~` is inversion. (True->False, False->True)

        cv2.imshow("video", frame)

    if writer.isOpened():
        writer.release()
    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()
```
- The following line in `video_recorder.py` is in order to open the VideoWriter for preparing the video-image recording.
    ```python
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"H264"), fps, (int(wt), int(ht)))
    ```
    - `cv2.VideoWriter_fourcc(*"H264")` specifies the codec of the recording. You can use the H264 codec if there is a codec file in the code folder.
    - You can see the arguments and more details of `VideoWriter()` [here](https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html)
- It can be started to record the video-image to `video_name` file when the user presses `r` key. And its recording is stopped when the user presses `r` key again. It's implemented with the value in `recflag` flag.
- The following line in `video_recorder.py` is in order to write the frame on the file.
    ```python
    writer.write(frame)
    ```
- The following line in `video_recorder.py` is in order to close the writer before the program is finished.
    ```python
    writer.release()
    ```
- You can find `video_name` in the `ipbl24` folder.


---
[back to the top page](../README.md)

