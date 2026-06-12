# Image processing basics for video image (my_av2.py)

[back to the top page](../README.md)

---

## Objectives
- This page explains how to process the video image in Python 3 with high-precision time management.

## Prerequisite
- Open the VSCode by the running the `py26en_start` on the Desktop. Confirm that the current directory shown in the terminal window is `C:\oit\home\ipbl`.
- **[CRITICAL]** Make sure that `my_av2.py` is placed in your `C:\oit\home\ipbl\my_libs` folder. This custom library is used for handling precise timestamps in real-time processing and video playback using PyAV library.
- The python program (.py) has to be made in `C:\oit\home\ipbl` folder. And all image files are saved (downloaded) in `C:\oit\home\ipbl\img` folder and read from there.
- You can run a python program with the input of the following command in the terminal.
    ```sh
    C:\oit\home\ipbl> python XXX.py
    ```

## :red_square: Sample of high-precision video-image processing

When using standard OpenCV (`cv2.VideoCapture`), processing delays (such as heavy AI model inference) can cause internal buffer accumulation, resulting in the display of past frames rather than real-time ones. Furthermore, when playing VFR (Variable Frame Rate) videos, standard OpenCV often fails to maintain the correct playback speed, causing the video to appear accelerated or slowed down.

To solve these issues, we use a custom `VideoCapture` and `VideoWriter` provided in `my_av2.py`. 

The class design is **fully compatible with OpenCV**. You can upgrade from standard OpenCV to this high-precision version simply by importing it from `my_libs.my_av2` and removing the `cv2.` prefix from the instance initialization.

### video_viewer1.py
```python
import os
# [https://github.com/opencv/opencv/issues/17687](https://github.com/opencv/opencv/issues/17687)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
# Import the high-precision VideoCapture from custom library
from my_libs.my_av2 import VideoCapture

device = 0 # camera device number

# main----------------------------------------------------
def main():
    global device

    # Use custom VideoCapture instead of cv2.VideoCapture(device)
    cap = VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)
    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current time position in milliseconds
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        # print("Current Time:", current_msec, "ms")

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
- A value of global variable `device` in `video_viewer1.py` is the device number of the camera starting from 0.
    - It can read a movie file when a value of the `device` is set to a movie file name, like the following.
        ```python
        device = "./img/moviefile.mp4"
        ```
- The following codes in `video_viewer1.py` are used to open the video stream and get the properties of the video image.
    | code | comment |
    | :--- | :--- |
    | `VideoCapture(device)` | Open the precision video stream (from `my_libs.my_av2`) |
    | `cap.get(cv2.CAP_PROP_FPS)` | FPS (frame / sec) rate |
    | `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | Frame width |
    | `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` | Frame height |
    | `cap.get(cv2.CAP_PROP_POS_MSEC)` | Current position of the video file or elapsed time of the camera in milliseconds |

### Read a frame from video stream
- The following function in `video_viewer1.py` is used to read a frame from the video stream.
    | code | comment |
    | :--- | :--- |
    | `cap.read()` | 1st return value is a boolean value for whether success in reading a frame <br> 2nd return value is the list of the pixel values in a frame |
    - Unlike standard OpenCV, `cap.read()` in `my_libs.my_av2` handles the frame acquisition based on the precise PTS (Presentation Time Stamp). 
    - **In camera (real-time stream):** Since the camera device and the program thread run asynchronously, frame drops are natural. `cv2.waitKey(1)` is ideal because it allows the loop to retrieve the newest available frame instantly, throwing away older unneeded frames to ensure pure real-time processing.
    - **In movie file playback:** `cap.read()` blocks the stream until the next frame is ready. It processes all frames sequentially without any drops, and `cap.get(cv2.CAP_PROP_POS_MSEC)` guarantees the exact original recorded time.

### Wait for the user's key input
- `cv2.waitKey()` function sleeps the process (thread) in order to wait for the user's key input during a value of the argument (m sec).
- It exits the while loop when the user presses the `q` key.

### :o:Practice
- Copy [`video_viewer1.py`](#video_viewer1py), paste it to VS Code, and save it as `video_viewer1.py` in the `C:\oit\home\ipbl` folder.
- If there is only one camera on your device, including the built-in, the number of your camera device is `0`.
- Run the sample code and verify that the video window comes up properly and terminates when you press the `q` key.

---

## :red_square: Exercise (`selfie.py`)
- Make a "Let's selfie program" (`selfie.py`) by modifying [`video_viewer1.py`](#video_viewer1py).
- Save a video frame to a still image file at the time when the user presses the `s` key.
    | Key | Details |
    | :---: | :--- |
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
  - The following function in the hint code is used to write a still image.
    | code | comment |
    | :--- | :--- |
    | `cv2.imwrite("name", variable)` | 1st argument is the file name(path) of the image which is saved. <br>2nd argument is the variable of the image. |

  - The following function in the hint code is used to compare a received key with a key that you want to detect.
    | code | comment |
    | :--- | :--- |
    | `ord('a character')` | It changes a character in the argument to the number of Unicode. |

- If your program is correct, you will find a jpeg file named `selfie.jpg` in the `img` folder when you press the `s` key.

---

## :red_square: Appendix: Sample of video-image recorder with precision log

The custom library `my_libs.my_av2` also provides a `VideoWriter` class that can record video while automatically generating a `.log.csv` file containing theoretical milliseconds, actual milliseconds, and lag delays.

### video_recorder.py
```python
# Sample of video-image recorder
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
# Import both precision VideoCapture and VideoWriter from custom library
from my_libs.my_av2 import VideoCapture, VideoWriter

device = 0 # camera device number
video_name = "record.mp4"

# main----------------------------------------------------
def main():
    global device, video_name
    recflag = False

    cap = VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    # Use precision VideoWriter instead of cv2.VideoWriter
    writer = VideoWriter(video_name, fps, (int(wt), int(ht)))

    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret:
            break
            
        if recflag:
            writer.write(frame)
            cv2.circle(frame, (20, 20), 5, [0,0,255], -1)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            recflag = not recflag # Inversion of the flag

        cv2.imshow("video", frame)

    writer.release()
    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()
```
- The following line in `video_recorder.py` is used to open the VideoWriter for preparing the video-image recording.
    ```python
    writer = VideoWriter(video_name, fps, (int(wt), int(ht)))
    ```
    - It uses `my_av2.py (PyAV)` internally to encode H.264 video. You do not need to specify `fourcc` codes manually.
- It starts recording the video-image to the `video_name` file when the user presses the `r` key, and stops when pressed again.
- When `writer.release()` is called, it automatically creates a `record.log.csv` next to the video file, which is useful for analyzing processing lag and frame drops.

---
[back to the top page](../README.md)
