# Image processing basics for static images

[back to the top page](../README.md)

---

## Objectives
- This page explains basics of digital images and image processing with Python3.

## prerequisite
- "[Image Processing Environment for iPBL](https://github.com/ipbl-oit-siit/portal/blob/main/setup/python%2Bvscode.md)" has already been installed.
- The python programs (.py) have to be put under the directory `SourceCode`. And the all image files are saved/downloaded in the directory `image` and read from there.

## :green_square: Basics of digital images
### :red_square: Color (Additive color)
- Many colors can be created by mixing the primary colors (Blue, Green, Red).<br>
    <image src="../image/Color-additive-mixing.png" height=25% width=25%><br>
    Additive color mixing([wikipedia](https://commons.wikimedia.org/wiki/File:Color-additive-mixing.png))

### :red_square: Digital images
- Digital images consist of many pixels. Pixel is the smallest unit in image space.
- Each pixel has a color (three values: Blue, Green, Red).<br>
    <image src="../image/pixels.png"><br>
    Digital image & pixels

### :red_square: Data structure of digital images
- Digital color images can be represented by 3 dimensional array.<br>
    <image src="../image/imageArray.png" height=30% width=30%><br>
    Color image array
- Range of pixel value is `0` to `255` (8bit). Thus, each pixel can create 16,777,216 (`=(256)^3`) colors.


## :green_square: Image processing with Python3
### :red_square: Directory structure for Image processing
- The python programs (.py) have to be put under the directory `SourceCode`. And the all image files are saved/downloaded in the directory `image` and read from there.
- Directory stucture
    ```text
    +[SourceCode]         <== work directory ("C:\oit\py23_ipbl\SourceCode")
    |
    |-+[image]            <== this directory already exists.
    | |--xxx.jpg
    | |--xxx.png
    | |--xxx.bmp
    | |--...              <== save new image files at this place.
    | |
    | |-+[standard]
    | | |--Aerial.bmp
    | | |--...
    | | |--Mandrill.png   <== this image already exists.("C:\oit\py23_ipbl\SourceCode\image\standard\Mandrill.bmp")
    | | |--...
    | | |
    | | |-+[mono]
    | | | |--Airplane.bmp
    | | | |--...
    |
    |-+[samples]
    | |--hello_python.py  <== this program already exists. ("C:/oit/py23_ipbl/SourceCode/test.py")
    | |--show_image.py    <== this program already exists.
    |
    |--python files(.py)  <== save new python programs at here.
    |--MediapipeFaceDetection.py
    |--MediapipeFaceLandmark.py
    |--...
    |--myhand.py
    |--...
    ```

### :red_square: Basics of Python3 program
- more information: [python3.10 docs](https://docs.python.org/3.10/index.html)
- Indentation is very important in Python programming. Indentation level is used to determine the coding block (the grouping of statements) and scope of variables.
- Variable is accessible from same block or nested block. Variable doesn't need to declare before using. Type of variable is determined by value to be assigned.Variable declared "global" has globally scope.
- A comment starts with a hash character `#`<br>
    <image src="../image/pys.png"><br>

#### prerequisite
- Open the VSCode by the running the `py23i_start.bat`. Confirm that the current directory shown in the terminal window is `SourceCode`.
- The python program (.py) has to be made in `SourceCode` folder. And all image files are saved (downloaded) in `image` folder and read from there.
- You can run a python program with the input of the following command in the terminal.
    ```sh
    C:\\...\SourceCode> python XXX.py
    ```
#### :o:Practice[basic]
- Save the following sample code as a python file and execute it. (`C:/oit/py23_ipbl/SourceCode/sample_basic.py`)
    <image src="../image/file_sample_basic.jpg" width=50%, height=50%>
- `sample_basic.py`
    ```python
    sum = 0
    for i in range(10):
        sum = sum + i
        print(str(i) + ":" + str(sum))

    if sum <= 30 :
        print("sum is under 30")
    elif sum <= 50 :
        print("sum is between 30 and 50")
    else:
        print("sum is over 50")
    ```
- It is O.K., if it is executed as follows.
  ```sh
  C:\\...\SourceCode> python sample_basic.py
  0:0
  1:1
  2:3
  3:6
  4:10
  5:15
  6:21
  7:28
  8:36
  9:45
  sum is between 30 and 50
  ```

### :red_square: Important modules in image processing
#### :blue_square: `numpy` (short name: `np`)
- more information: [numpy docs](https://numpy.org/doc/stable/)
- This module is the fundamental package for scientific computing.
    - a powerful `N`-dimensional array object
    - useful linear algebra, Fourier transform, and random number capabilities

#### :o:Practice[np]
- Save the following sample code as a python file and execute it. (`C:/oit/py23_ipbl/SourceCode/sample_numpy.py`)
- `sample_numpy.py`
    ```python
    import numpy as np
    a = np.zeros((4, 3, 2))  # make zero array whose size is (4,3,2)
    a[0:2, 1:2, 1] = 1  # Note that, 0:2 means 0 to (2-1), and 1:2 means 1.
    print(a)
    print(np.average(a))
    print(np.max(a))
    ```
- It is O.K., if it is executed as follows.
    ```sh
    C:\\...\SourceCode> python sample_numpy.py
    [[[0. 0.]
        [0. 1.]
        [0. 0.]]

    [[0. 0.]
        [0. 1.]
        [0. 0.]]

    [[0. 0.]
        [0. 0.]
        [0. 0.]]

    [[0. 0.]
        [0. 0.]
        [0. 0.]]]
    0.08333333333333333
    1.0
    ```

#### :blue_square: `cv2` (opencv-python)
- more information: [OpenCV4.7.0 docs](https://docs.opencv.org/4.7.0/)
- This is an open source module for Computer Vision.
- It has many functions for image processing.

#### :o:Practice[cv2]
- Save the following sample code as a python file, and execute it. (`C:/oit/py23_ipbl/SourceCode/sample_cv2.py`)
- `sample_cv2.py`
    ```python
    import cv2
    img = cv2.imread('./image/standard/Mandrill.bmp') # read image file
    if img is None: # maybe Path is wrong
        print('ERROR: image file is not opened.')
        exit(1)
    bimg = cv2.GaussianBlur(img, (51,51), 5) # gaussian filter (size=(51,51),sigma=5)
    cv2.imshow('img',img)
    cv2.imshow('blur img',bimg)
    cv2.waitKey(0) # pause until press any key
    cv2.destroyAllWindows # close all cv2's windows
    ```
- It is O.K., if the following windows pop up.<br>
  <image src="../image/Mandrill_blur.png" height=50% width=50%>
- The windows close when you press any key.

### :red_square: Script/Function in Python3 and image IO
- Making a Python script a function improves reusability.
  - Functions can be called by other python programs.

#### Python Script `sample_imgIO.py`
    ```python
    import cv2

    # read image file
    img = cv2.imread('./image/standard/Mandrill.bmp')
    if img is None:
        print('ERROR: image file is not opened.')
        exit(1)

    # write image file
    cv2.imwrite('./image/res_scrpt.png', img)

    # show image file
    cv2.imshow('window name', img)
    cv2.waitKey(0)  # pause until any key pressed
    cv2.destroyAllWindows()  # close all windows
    ```

#### Python Function `sample_imgIO_func.py`
    ```python
    import cv2
    a = 1 # global variable

    def imageIO(img_name_in, img_name_out):
        # read image file
        img = cv2.imread(img_name_in)
        if img is None:
            print('ERROR: image file is not opened.')
            exit(1)

        # write image file
        cv2.imwrite(img_name_out, img)
        return img

    def main():
        print(a, b) # print global variables
        in_name = './image/standard/Mandrill.bmp' # local variable
        out_name = './image/res_func1.png' # local variable
        img = imageIO(in_name, out_name)
        # show image file
        cv2.imshow('window name', img)
        cv2.waitKey(0)  # pause until any key pressed
        cv2.destroyAllWindows()  # close all windows

    # The following equation holds when this program file is only executed.
    if __name__ == '__main__':
        b = 0 # global variable
        main() # function name is free
    ```

#### :o:Practice[script/function 1]
- Save the above two sample codes (`sample_imgIO.py`, `sample_imgIO_func.py`) as a python file. (`C:/oit/py23_ipbl/SourceCode/sample_imgIO.py`) (`C:/oit/py23_ipbl/SourceCode/sample_imgIO_func.py`)
- Execute the two python codes, respectively.
- It's O.K., if the two result images (`res_scrpt.png`, `res_func1.png`) in the directory `image` are the same.

#### :o:Practice[script/function 2]
- Let's use the function `imageIO` in `sample_imgIO_func.py` on Other python programs!
- After `Practice[script/function 1]`, Save the following sample code as a python file, and execute it. (`C:/oit/py23_ipbl/SourceCode/sample_other.py`)
- `sample_other.py`
  ```python
  import sample_imgIO_func as myImgIO

  myImgIO.imageIO('./image/standard/Mandrill.bmp', './image/res_func2.png')
  ```
- It's O.K., if the all result images (`res_scrpt.png`, `res_func1.png`, `res_func2.png`) in the directory `image` are the same.

### :red_square: Python Code for Resizing images
- The process with large size images is very heavy. If image size is huge, you should resize it to small.
- There are various methods for resizing.
    - Resizing with specified size
        ```python
        # the size of img_resize becomes (new_width, new_height).
        img_resize = cv2.resize(img, (new_width, new_height))
        ```
    - Resizing with scalling
        ```python
        # downscalling -> 1/2
        img_resize = cv2.resize(img, None, fx=1/2, fy=1/2)
        ```
    - Resizing the long side of images to a specified length while keeping the aspect ratio
        - This program can resize Images of various sizes to approximately the same data size while maintaining the aspect ratio.
        ```python
        def resizeImg(img, length):
            h, w = img.shape[:2]

            if max(h, w) < length: # do not need resizing
                return img

            if h < w:
                newSize = (int(h*length/w), length)
            else:
                newSize = (length, int(w*length/h))

            print('resize to', newSize)

            return cv2.resize(img, (newSize[1], newSize[0])) # (w, h)
        ```

#### :o:Exercise[resizing]
- Please edit `resize.py` and type the following template. It's O.K. copy and paste.
    ```python
    import cv2

    def resizeImg(img, length):
        """
        This function resizes the long side of images to the specified length while keeping the aspect ratio.

        Args:
            img(numpy.ndarray): input image
            length(int): length of long side after resizing

        Returns:
            numpy.ndarray: resized image
        """
        h, w = img.shape[:2]
        if max(h, w) < length:
            return img
        if h < w:
            newSize = (int(h*length/w), length)
        else:
            newSize = (length, int(w*length/h))
        print('resize to', newSize)
        return cv2.resize(img, (newSize[1], newSize[0])) # (w, h)

    def main():
        in_name = './image/standard/Mandrill.bmp'
        img = cv2.imread(in_name)
        if img is None:
            print('ERROR: image file is not opened.')
            exit(1)

        img150x100 = cv2.resize(img, (150, 100))
        img_half = cv2.resize(img, None, fx=2/3, fy=2/3)
        img150 = resizeImg(img.copy(), 150)

        cv2.imshow('img', img)
        cv2.imshow('img150x100', img150x100)
        cv2.imshow('img_half', img_half)
        cv2.imshow('img150', img150)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if __name__ == '__main__':
        main()
    ```
- Please run `resize.py`.
- It's O.K., if the following figures pops up.<br>
    <image src="../image/Mandrill_resizing.png" height=50% width=50%>

---

[back to the top page](../README.md)
