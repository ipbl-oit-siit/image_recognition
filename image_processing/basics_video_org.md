# Image processing basics for video image

[back to the top page](../README.md)

---

## Objectives
- This page explains how to process the video image in Python 3 with high-precision time management.

## Prerequisite
- Open the VSCode by the running the `py26en_start` on the Desktop. Confirm that the current directory shown in the terminal window is `C:\oit\home\ipbl`.
- **[CRITICAL]** Make sure that `my_av2.py` is placed in your `C:\oit\home\ipbl\my_libs` folder. This custom library is used for handling precise timestamps in real-time processing and video playback using PyAV library.
- All image files are saved (downloaded) in `C:\oit\home\ipbl\img` folder and read from there.
- You can run a python program with the input of the following command in the terminal.
    ```sh
    C:\\oit\home\ipbl> python XXX.py
    ```

## :red_square: Sample of high-precision video-image processing

When using standard OpenCV (`cv2.VideoCapture`), processing delays (such as heavy AI model inference) can cause internal buffer accumulation, resulting in the display of past frames rather than real-time ones. Furthermore, when playing VFR (Variable Frame Rate) videos, standard OpenCV often fails to maintain the correct playback speed, causing the video to appear accelerated or slowed down.

To solve these issues, we use a custom `VideoCapture` and `VideoWriter` provided in `my_av2.py`. 

The class design is **fully compatible with OpenCV**. You can upgrade from standard OpenCV to this high-precision version simply by importing it from `my_av2` and removing the `cv2.` prefix from the instance initialization.

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
