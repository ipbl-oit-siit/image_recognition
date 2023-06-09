# Resizing images

[README](../README.md)

---

## Objectives

This page explains how to resize images.

## Prerequisite

You have to finish [Image processing basics for static images](../basics/basics_static.md).

## Python Code for Resizing images
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

## :o:Exercise[resizing]
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
    in_name = './img/Mandrill.png'
    img = cv2.imread(in_name)

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

### Run
- Please run `resize.py`.
- It's O.K., if the following figures pops up.<br>
  <image src="../image/Mandrill_resizing.png" height=50% width=50%>

---

[README](../README.md)
