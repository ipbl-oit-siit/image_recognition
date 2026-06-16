# Hula-JP Python(3.12.10) Control & Image Analysis

This document is a practical hands-on guide for aircraft control and image analysis using the **Hula-JP – Python (3.12.10) Installation Package**.

---

## ■ Practice 1: Communication
Establish a UDP connection with the aircraft or control board.

```python
import socket

TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889
CONTROL_ADDRESS = (TELLO_IP, TELLO_PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 9000))

```

### 〇 Explanation

* **`socket.SOCK_DGRAM`**: Creates a UDP socket used for fast, low-overhead communication with the drone.
* **`sock.bind`**: Binds the local port `9000` to listen for status responses sent back from the aircraft.

---

## ■ Practice 2: Initialization (SDK Mode)

Activate the SDK mode to make the aircraft ready to accept further control commands.

```python
def initialize_drone():
    send_command("command")
    send_command("battery?")

```

### 〇 Explanation

* **`command`**: Instructs the drone to enter its remote control / SDK mode.
* **`battery?`**: Requests the current battery percentage to ensure the aircraft is safe to operate.

---

## ■ Practice 3: Main Program Template with Integrated Failsafes

A robust boilerplate framework that guarantees a safe landing (`finally` block) even if the script encounters errors or user interruptions (`Ctrl+C`).

```python
import socket
import sys
import time

def send_command(command: str):
    try:
        print(f"[TX]: {command}")
        sock.sendto(command.encode('utf-8'), CONTROL_ADDRESS)
        
        sock.settimeout(3.0)
        data, _ = sock.recvfrom(1518)
        response = data.decode('utf-8').strip()
        print(f"[RX]: {response}")
        return response
        
    except socket.timeout:
        print("\n[!!! COM LOSS DETECTED !!!] Aborting program.")
        sys.exit(1)

def main():
    send_command("command")
    
    battery = send_command("battery?")
    try:
        if int(battery) <= 20:
            print("[ERROR] Battery too low. Aborting takeoff.")
            return
    except ValueError:
        print("[WARNING] Could not parse battery level.")

    try:
        print("\n--- Starting Flight Sequence ---")
        
        # --- WRITE YOUR FLIGHT COMMANDS HERE ---
        send_command("takeoff")
        time.sleep(5)
        
        send_command("up 50")
        time.sleep(4)
        
        send_command("forward 60")
        time.sleep(4)
        
        send_command("cw 90")
        time.sleep(4)
        # --------------------------------------
        
    except KeyboardInterrupt:
        print("\n[USER INTERRUPT] Program stopped by user.")
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        
    finally:
        print("\n[SAFETY] Sending LAND command.")
        try:
            send_command("land") 
        except Exception:
            print("Failed to send land command. Recover manually.")
            
        print("Program terminated safely.")

if __name__ == "__main__":
    main()

```

### 〇 Explanation

* **`sock.settimeout(3.0)`**: If the aircraft does not respond within 3 seconds, a `socket.timeout` exception is raised to handle communication loss immediately.
* **`try...finally`**: Ensures that the `land` command is executed unconditionally at the end of the script, preventing the drone from getting stuck mid-air.

---

## ■ Practice 4: Flight Motion

Basic moving commands and their parameters.

### 〇 Motion Rules & Limits

* **Distance / Altitude**: Specified in **`cm` (Centimeters)**. Valid range is **`20` to `500**`.
* **Rotation Angle**: Specified in **`Degrees` (°)**. Valid range is **`1` to `360**`.

| Command | Action | Unit / Range |
| --- | --- | --- |
| `takeoff` | Automatic Takeoff | None (Climbs to ~1m and hovers) |
| `land` | Automatic Landing | None (Descends and stops motors) |
| `up X` | Ascend | cm (20 to 500) |
| `down X` | Descend | cm (20 to 500) |
| `forward X` | Move Forward | cm (20 to 500) |
| `back X` | Move Backward | cm (20 to 500) |
| `left X` | Move Left | cm (20 to 500) *Maintains heading* |
| `right X` | Move Right | cm (20 to 500) *Maintains heading* |
| `cw X` | Clockwise Turn | Degrees (1 to 360) |
| `ccw X` | Counter-Clockwise | Degrees (1 to 360) |

---

## ■ Practice 5: Frame Acquisition

Capture a single real-time frame using the high-performance `pyav` backend.

```python
import cv2
from hula_camera import PyAVVideoCapture

def get_camera_frame():
    cap = PyAVVideoCapture(0)
    if not cap.is_opened():
        print("Failed to open camera pipeline")
        return None

    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

```

### 〇 Explanation

* **`PyAVVideoCapture`**: Hula-JP's hardware-accelerated wrapper that decodes video streams with minimal latency compared to standard OpenCV.

---

## ■ Practice 6: HSV Color Detection

Isolate target colors by converting the frame into the HSV color space.

```python
import cv2
import numpy as np

def detect_hsv_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([75, 255, 255])
    
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    result_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return mask, result_frame

```

### 〇 Explanation

* **`cv2.cvtColor`**: Converts the default BGR format to HSV (Hue, Saturation, Value), which is more stable for color detection under varying lights.
* **`cv2.inRange`**: Creates a binary mask where the pixels matching the green range turn white, and all other pixels turn black.

---

## ■ Practice 7: AR Marker Recognition

Detect ArUco markers to retrieve unique IDs and coordinate data for alignment.

```python
import cv2

def detect_ar_markers(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            print(f"Detected AR Marker ID: {ids[i][0]}")
            
    return frame, ids

```

### 〇 Explanation

* **`DICT_4X4_50`**: Specifies the internal ArUco dictionary standard (4x4 matrix grid up to 50 unique IDs).
* **`detectMarkers`**: Returns the corner coordinates and the marker IDs found within the current image frame.

---

## ■ Practice 8: Advanced Failsafes

Modular functions to prevent crashes and safely handle flight anomalies.

```python
def check_battery_safety(current_battery: int):
    LOW_BATTERY_THRES = 20      
    CRITICAL_BATTERY_THRES = 10 
    
    if current_battery <= CRITICAL_BATTERY_THRES:
        print("[CRITICAL] Battery critical. Forcing immediate landing.")
        send_command("land")
        return False
    elif current_battery <= LOW_BATTERY_THRES:
        print("[WARNING] Battery low. Terminating mission and landing safely.")
        send_command("land")
        return False
    return True

def emergency_stop():
    print("[!!! EMERGENCY !!!] Forcing immediate motor shutdown.")
    send_command("emergency")

```

### 〇 Explanation

* **`check_battery_safety`**: Evaluates the battery state; returns `False` if it is unsafe to continue the flight mission.
* **`emergency` command**: Instantly kills power to all rotors. *Note: The drone will drop immediately. Use only as a last resort to avoid human injury.*

```
