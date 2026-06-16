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

##### 2. Complete the Trigger Program (`ipB_time_trigger.py`)

* Open the distributed `ipB_time_trigger.py` file.
* **Complete the `TODO` sections** to calculate the diagonal size of the marker, filter for ID 2, and update the timer.

```python
import cv2
import numpy as np
from my_libs.my_av2 import VideoCapture
from my_libs.my_timer import DetectionTimer 

def main():
    cap = VideoCapture(0)
    
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

[back to the top page](../README.md)
