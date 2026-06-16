# Image processing and control for Hula-JP Drone

[back to the top page](../README.md)

---

## Objectives
- This page explains basics of Hula-JP drone control and real-time image analysis with Python3.

## prerequisite
- "[Python Environment for iPBL26](https://github.com/ipbl-oit-siit/portal/blob/main/setup/python%2Bvscode.md)" has already been installed.
- The python programs (.py) have to be put under the directory `C:\oit\py26\ipbl`. 
- The custom communication and video library `my_av2.py` must be located under the directory `mylibs`.

---

## :green_square: Communication & Initialization
### :red_square: Core Network Settings
- Establish a connection with the aircraft using the standard IP and Port configurations.

#### :o:Practice[communication]
- Save the following sample code as a python file to verify the network parameters. (`C:\oit\home\ipbl\sample_hula_init.py`)
- `sample_hula_init.py`
    ```python
    import socket

    HULA_IP = "192.168.10.1"
    HULA_PORT = 8889
    CONTROL_ADDRESS = (HULA_IP, HULA_PORT)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 9000))
    ```

> [!NOTE]
> ### Explanation
> - **`socket.SOCK_DGRAM`**: Creates a UDP socket used for fast, low-overhead communication with the drone.
> - **`sock.bind`**: Binds the local port `9000` to listen for status responses sent back from the aircraft.

---

## :green_square: Flight Control & Safety
### :red_square: Main Control Template with Integrated Failsafes
- A robust program template that guarantees a safe landing (`finally` block) even if the script encounters errors or user interruptions (`Ctrl+C`).

#### :o:Practice[failsafe_template]
- Save the following sample code as a python file and execute it. (`C:\oit\home\ipbl\sample_main_failsafe.py`)
- `sample_main_failsafe.py`
    ```python
    import socket
    import sys
    import time

    HULA_IP = "192.168.10.1"
    HULA_PORT = 8889
    CONTROL_ADDRESS = (HULA_IP, HULA_PORT)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 9000))

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

> [!NOTE]
> ### Explanation
> - **`sock.settimeout(3.0)`**: If the aircraft does not respond within 3 seconds, a `socket.timeout` exception is raised to handle communication loss immediately.
> - **`finally`**: Ensures that the `land` command is executed unconditionally at the end of the script, preventing the drone from getting stuck mid-air.

---

### :red_square: Flight Motion Rules & Limits
- Basic moving commands and their parameters.
- **Distance / Altitude**: Specified in **`cm` (Centimeters)**. Valid range is **`20` to `500`**.
- **Rotation Angle**: Specified in **`Degrees` (°)**. Valid range is **`1` to `360`**.

| Command | Action | Unit / Range |
| :--- | :--- | :--- |
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

## :green_square: Real-Time Image Processing
### :red_square: Frame Acquisition via `my_av2`
- Capture a single real-time frame using the custom low-latency video library `my_av2.py`.

#### :o:Practice[video_capture]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_video.py`)
- `sample_hula_video.py`
    ```python
    import cv2
    from mylibs.my_av2 import VideoCapture # use custom library for iPBL26

    def get_camera_frame():
        cap = VideoCapture() # open camera stream from drone
        if not cap.is_opened():
            print("Failed to open camera pipeline")
            return None

        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    ```

> [!NOTE]
> ### Explanation
> - **`from mylibs.my_av2 import VideoCapture`**: Imports the custom video capture class designed to decode drone camera streams with minimal delay.

---

### :red_square: HSV Color Detection
- Isolate target colors by converting the frame into the HSV color space.

#### :o:Practice[hsv_filter]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_hsv.py`)
- `sample_hula_hsv.py`
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

> [!NOTE]
> ### Explanation
> - **`cv2.cvtColor`**: Converts the default BGR format to HSV (Hue, Saturation, Value), which is more stable for color detection under varying lights.
> - **`cv2.inRange`**: Creates a binary mask where the pixels matching the green range turn white, and all other pixels turn black.

---

### :red_square: AR Marker Recognition
- Detect ArUco markers to retrieve unique IDs and coordinate data for alignment.

#### :o:Practice[aruco_detection]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_aruco.py`)
- `sample_hula_aruco.py`
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

> [!NOTE]
> ### Explanation
> - **`DICT_4X4_50`**: Specifies the internal ArUco dictionary standard (4x4 matrix grid up to 50 unique IDs).
> - **`detectMarkers`**: Returns the corner coordinates and the marker IDs found within the current image frame.

---

### :red_square: Advanced Failsafes & Emergency Methods
- Modular functions to prevent crashes and safely handle flight anomalies based on battery levels.

#### :o:Practice[emergency_routines]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_safety.py`)
- `sample_hula_safety.py`
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

> [!NOTE]
> ### Explanation
> - **`emergency` command**: Instantly kills power to all rotors. *Note: The drone will drop immediately. Use only as a last resort to avoid human injury.*

---

[back to the top page](../README.md)
