# Advanced Image Processing: Image Processing with Camera

[back to the top page](../README.md)

---
### :red_square: Time-Based Target Verification

In actual drone control, triggering a mission (such as landing or dropping an item) the exact millisecond a marker enters the frame is highly unstable. Sensor noise or sudden movement might cause the camera to instantly lose track, leading to false triggers or interrupted routines. 

To resolve this, we use a time-based validation system: the drone only triggers the next action when a specific target is stably recognized for a predefined duration.

#### Features of `DetectionTimer`
* **Pre-installed in your `my_libs` folder**: You don't need to write the complex state-machine logic from scratch.
* **Chattering Filter (Grace Period)**: It includes a built-in `grace_ms=300.0` (0.3 seconds) buffer. If the camera loses the marker for a split second due to lighting or reflection, the timer **holds its progress** instead of instantly wiping out your accumulated countdown.

---

### :o:Exercise [Time-Based Target Verification]
* Let's complete a program that triggers an event when **Card ID: 2 (Diamond Ace)** is continuously detected for **3 seconds (3000ms)** at close range.

##### 1. Understanding the `DetectionTimer` API
You can instantiate the timer by setting your target duration. Every frame, you feed the detection status into the `.update()` method:

```python
from my_libs.my_timer import DetectionTimer

# 1. Initialize with target duration in milliseconds
timer = DetectionTimer(target_ms=3000.0)

# 2. Update every frame inside your while loop
# Returns True only if the condition has been met for the target duration
is_cleared = timer.update(is_detected, current_msec)

```

##### 2. Complete the Trigger Program (`imp_time_trigger.py`)

* Open the distributed `imp_time_trigger.py` file.
* **Complete the `TODO` sections** to calculate the diagonal size of the marker, filter for ID 2, and update the timer.

```python
import cv2
import numpy as np
from my_libs.my_av2 import VideoCapture
from my_libs.my_timer import DetectionTimer 

device = 0 # camera device number

def main():
    cap = VideoCapture(device)
    
    # Initialize the timer to require 3 seconds (3000ms) of stable detection
    mission_timer = DetectionTimer(target_ms=3000.0)
    
    # Initialize the ARuCo detector dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    print("Looking for Card ID: 2 (Diamond Ace)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # --- [Step 1: Image Processing] ---
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
        
        is_target_valid = False
        
        if ids is not None:
            flat_ids = ids.flatten()
            
            # Check if Card ID 2 (Diamond Ace) is inside the visible stream
            if 2 in flat_ids:
                # Extract index and corners for ID 2
                idx = np.where(flat_ids == 2)[0][0]
                marker_corners = corners[idx].reshape((4, 2))
                
                # TODO: Calculate diagonal distance (Top-Left [0] to Bottom-Right [2])
                size_px = np.linalg.norm(_________________ - _________________)
                
                # TODO: Condition is met if Card ID 2 is close enough (size > 150 pixels)
                if size_px > 150:
                    is_target_valid = True
                    
                # Visual feedback: Draw a border over the target marker
                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[2]]))

        # --- [Step 2: Time Verification] ---
        # TODO: Pass the detection status and current timestamp into the timer
        is_cleared = mission_timer.update(________________, ________________)

        # --- [Step 3: Visual Feedback and Drone Trigger] ---
        if is_cleared:
            cv2.putText(frame, "TARGET VERIFIED! TRIGGERING MISSION...", (30, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            # (In the next module, you will place drone flight commands here, e.g., drone.land())
        elif is_target_valid:
            cv2.putText(frame, "Target Found: Counting down...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Time-Based Trigger System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

* It is O.K., if you hold Card ID: 2 close to your camera, and a bright green "TARGET VERIFIED" alert locks onto the window screen after exactly 3 seconds of stable tracking.

---
### :red_square: Application: Tracking a Pink Box with the Same Logic

Next, let's look at how the exact same time-verification logic can be applied to **HSV Color Extraction**. 

Whether you are detecting a 2D ArUco marker or a physical colored object like a **Pink Box**, you can treat them identically by calculating their **diagonal size** in pixels. Once converted into a 1D diagonal size, Step 2 and Step 3 remain completely unchanged.

#### Geometry Difference: Marker vs. Color Contour
* **ArUco Marker**: Provides 4 explicit corner points. Diagonal is calculated as the distance between Top-Left `[0]` and Bottom-Right `[2]`.
* **Color Bounding Box**: Provides a rectangle with Width (`w`) and Height (`h`). Diagonal is calculated using the Pythagorean theorem: $\sqrt{w^2 + h^2}$.

---

### :o:Exercise 2 [Time-Based Color Verification]
* Let's complete another program that triggers an event when a **Pink Box** is stably detected for **3 seconds (3000ms)** at close range.

##### 1. Complete the Color Trigger Program (`imp_color_trigger.py`)
* Open the distributed `imp_color_trigger.py` file.
* **Complete the `TODO` sections** to calculate the diagonal length of the bounding box using `np.sqrt()`, and update the same timer structure.

```python
import cv2
import numpy as np
from my_libs.my_av2 import VideoCapture
from my_libs.my_timer import DetectionTimer 

device = 0 # camera device number

def main():
    cap = VideoCapture(device)
    
    # Initialize the timer to require 3 seconds (3000ms) of stable detection
    mission_timer = DetectionTimer(target_ms=3000.0)
    
    # Define HSV thresholds for Pink Box (Adjust based on your classroom lighting)
    LOWER_PINK = np.array([140,  50,  50])
    UPPER_PINK = np.array([175, 255, 255])

    print("Looking for the Pink Box...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # --- [Step 1: Image Processing (HSV Color Extraction)] ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_target_valid = False
        
        if len(contours) > 0:
            # Get the largest pink contour and its bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # TODO: Calculate diagonal distance of the bounding box using w and h
            size_px = np.sqrt(_________________ + _________________)
            
            # TODO: Condition is met if the Pink Box is close enough (size > 150 pixels)
            if size_px > 150:
                is_target_valid = True
            
            # Visual feedback: Draw a bounding box over the detected pink object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # --- [Step 2: Time Verification] ---
        # TODO: Pass the detection status and current timestamp into the timer
        is_cleared = mission_timer.update(________________, ________________)

        # --- [Step 3: Visual Feedback and Drone Trigger] ---
        if is_cleared:
            cv2.putText(frame, "TARGET VERIFIED! TRIGGERING MISSION...", (30, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        elif is_target_valid:
            cv2.putText(frame, "Target Found: Counting down...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Time-Based Trigger System (Color)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

* Show the Pink Box to your camera, and verify that the green "TARGET VERIFIED" text triggers stably after 3 seconds, just like the ARuCo marker task.

---

[back to the top page](../README.md)
