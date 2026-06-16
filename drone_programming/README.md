# Image processing and control for Hula-JP Drone

[back to the top page](../README.md)

---

## Objectives
- This page explains basics of Hula-JP drone control and real-time image analysis with Python3.

## prerequisite
- "[Python Environment for iPBL26](https://github.com/ipbl-oit-siit/portal/blob/main/setup/python%2Bvscode.md)" has already been installed.
- The python programs (.py) have to be put under the directory `C:\oit\py26\ipbl`. 
- The custom libraries `my_av2.py` and `detection_timer.py` must be located under the directory `mylibs`.

---

## :green_square: Pre-Flight Safety & Connection Tests
Before performing any actual flight sequence, always execute these non-takeoff tests to ensure safe hardware and video stream operations.

### :red_square: Step 1: Communication & Battery Status Test
- Establish network synchronization and retrieve the current battery level without starting the motors.

#### :o:Practice[ping_and_battery]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_ping.py`)
- `sample_hula_ping.py`
    ```python
    import socket
    import sys

    HULA_IP = "192.168.10.1"
    HULA_PORT = 8889
    CONTROL_ADDRESS = (HULA_IP, HULA_PORT)

    # Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 9000))

    def send_test_command(command: str):
        try:
            print(f"[TX]: {command}")
            sock.sendto(command.encode('utf-8'), CONTROL_ADDRESS)
            
            sock.settimeout(3.0)
            data, _ = sock.recvfrom(1518)
            response = data.decode('utf-8').strip()
            print(f"[RX]: {response}")
            return response
        except socket.timeout:
            print("[ERROR] Connection timeout. Check Wi-Fi connection to the drone.")
            sys.exit(1)

    def main():
        print("--- Initiating Drone Communication Test ---")
        # 1. Enter SDK mode
        send_test_command("command")
        
        # 2. Query Battery Capacity
        battery = send_test_command("battery?")
        print(f"\n[STATUS] Connection successful. Battery Level: {battery}%")

    if __name__ == "__main__":
        main()
    ```

> [!NOTE]
> ### Explanation
> - **`command`**: Instructs the drone to enter its automated SDK control state.
> - **`battery?`**: Queries the internal telemetry block. Returns an integer string from `0` to `100`.

---

### :red_square: Step 2: Ground Motor Rotation Test (No Takeoff)
- Spin the propellers at a low idle speed on the ground to check motor status without generating lift.

#### :o:Practice[motor_test]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_motor_test.py`)
- `sample_hula_motor_test.py`
    ```python
    import socket
    import time
    import sys

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
            print("[!!! EMERGENCY !!!] Lost connection during motor test.")
            sys.exit(1)

    def main():
        send_command("command")
        
        try:
            print("\n--- Starting Propeller Rotation Test ---")
            # Spin the motors at idle speed on the ground
            send_command("motoron")
            print("Motors spinning at idle speed... checking hardware status.")
            time.sleep(3)
            
            # Turn off the motors immediately
            send_command("motoroff")
            print("Motors stopped safely.")
            
        except KeyboardInterrupt:
            print("\n[USER INTERRUPT] Stopping motors immediately.")
            send_command("motoroff")

    if __name__ == "__main__":
        main()
    ```

> [!NOTE]
> ### Explanation
> - **`motoron`**: Starts all four propulsion modules at a minimal idle rate. The aircraft will remain firmly on the ground.
> - **`motoroff`**: Instantly cuts off the motor rotation queue for safety preservation.

---

### :red_square: Step 3: Ground Camera Stream Test (No Takeoff)
- Verify the video pipeline and latency by streaming the camera feed to an OpenCV window while the drone stays securely on the ground.

#### :o:Practice[stream_test]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_stream_test.py`)
- `sample_hula_stream_test.py`
    ```python
    import cv2
    import socket
    from mylibs.my_av2 import VideoCapture

    HULA_IP = "192.168.10.1"
    HULA_PORT = 8889
    CONTROL_ADDRESS = (HULA_IP, HULA_PORT)

    def main():
        # 1. Initialize control socket and enable video command
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", 9000))
        
        print("[TX]: command")
        sock.sendto(b"command", CONTROL_ADDRESS)
        
        print("[TX]: streamon")
        sock.sendto(b"streamon", CONTROL_ADDRESS)

        # 2. Connect to the custom PyAV video pipeline
        video_source = 'udp://0.0.0.0:11111'
        cap = VideoCapture(video_source)

        if not cap.isOpened():
            print("[ERROR] Cannot open drone video stream.")
            return

        print("\n--- Video Stream Started ---")
        print("Press 'q' inside the video window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame dropped.")
                continue

            # Display the live frame
            cv2.imshow("Hula-JP Ground Camera Test", frame)

            # Safely exit loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing video stream...")
                break

        # 3. Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released successfully.")

    if __name__ == "__main__":
        main()
    ```

> [!NOTE]
> ### Explanation
> - **`streamon`**: Commands the drone backend to start broadcasting its video frames via UDP port 11111.
> - **`cv2.waitKey(1) & 0xFF == ord('q')`**: Monitors keyboard events every millisecond. Intercepts character comparisons to provide an intentional, non-crash exit path.

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
- Capture a real-time frame by passing the video source into the custom OpenCV-compatible `VideoCapture` class.

#### :o:Practice[video_capture]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_video.py`)
- `sample_hula_video.py`
    ```python
    import cv2
    from mylibs.my_av2 import VideoCapture # use custom library for iPBL26

    def get_camera_frame():
        # Pass a camera index (int) or a streaming URL string (str) as the source
        # Example: 0 for local camera, 'udp://0.0.0.0:11111' for drone video stream
        video_source = 'udp://0.0.0.0:11111'
        cap = VideoCapture(video_source) 
        
        if not cap.isOpened():
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
> - **`VideoCapture(video_source)`**: Dynamically initializes the stream container depending on the variable type. It accepts an integer for OpenCV webcam streaming, or a string for PTS-based accurate VFR video decoding via PyAV.

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
- Detect specialized ArUco markers printed for the Hula-JP environment to retrieve localization values.

#### :o:Practice[aruco_detection]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_aruco.py`)
- `sample_hula_aruco.py`
    ```python
    import cv2

    def detect_ar_markers(frame):
        # CHANGE HERE: Use the specific ArUco dictionary specified by your Hula-JP task sheet
        # Example: cv2.aruco.DICT_6X6_250, cv2.aruco.DICT_APRILTAG_36h11, etc.
        HULA_AR_DICTIONARY = cv2.aruco.DICT_4X4_50 
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(HULA_AR_DICTIONARY)
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
> - **`HULA_AR_DICTIONARY`**: Defines the layout matrix configuration. **Make sure to change `DICT_4X4_50` to the specific dictionary format designated in your course handbook**, otherwise the tracking grid will mismatch and ignore physical markers.
> - **`detectMarkers`**: Returns the corner coordinates and the marker IDs found within the current image frame.

---

## :green_square: State Tracking & Decision Making
### :red_square: Real-Time Stabilization Loop via `cv2.waitKey` & `DetectionTimer`
- **Critical Requirement**: To control the drone safely in real time without video streaming lag, your main loop must run **completely non-blocking**.
- Using `time.sleep()` inside the loop will cause the UDP video packet buffer to overflow, resulting in severe 2-3 second video lag and application freezes. 
- You must ingest frames continuously using `VideoCapture` and control the iteration rate using **`cv2.waitKey(1)`**.

#### :o:Practice[detection_timer]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_timer.py`)
- `sample_hula_timer.py`
    ```python
    import cv2
    from mylibs.my_av2 import VideoCapture
    from mylibs.detection_timer import DetectionTimer

    def track_target_mission(duration_threshold=3.0):
        # 1. Initialize custom timer and connect to the drone stream
        timer = DetectionTimer(target_seconds=duration_threshold)
        
        video_source = 'udp://0.0.0.0:11111'
        cap = VideoCapture(video_source)
        
        if not cap.isOpened():
            print("[ERROR] Cannot connect to drone video stream.")
            return

        print("Starting real-time non-blocking detection loop...")
        print("Press 'q' in the window to abort.")

        try:
            while True:
                # 2. Ingest the latest frame immediately without blocking time
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If frame drop occurs, yield execution instantly to keep loop spinning
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # 3. Target evaluation placeholder
                # Replace True with actual vision logic component (e.g., 'ids is not None')
                is_detected = True 
                
                # 4. Update the tracking metrics on a frame-by-frame basis
                is_stable, elapsed = timer.update(is_detected)
                
                # Render HUD feedback onto the frame
                status_text = f"Lock: {elapsed:.1f}s / Target Stable: {is_stable}"
                color = (0, 255, 0) if is_stable else (0, 0, 255)
                cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Real-Time Tracking Window", frame)
                
                # 5. If verified stably for 3.0 seconds, break to execute next control command
                if is_stable:
                    print(f"\n[SUCCESS] Target verified stably for {duration_threshold}s!")
                    break
                    
                # 6. Use minimal 1ms non-blocking wait to refresh GUI and check keyboard abort
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[USER ABORT] Mission interrupted.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    ```

> [!NOTE]
> ### Explanation
> - **`cap.read()` inside a fast loop**: Frees the network buffer continuously, ensuring the frame processed is always a "fresh live frame" rather than an old backlogged buffer.
> - **`cv2.waitKey(1)`**: Replaces `time.sleep()`. It yields CPU execution for exactly 1 millisecond to handle internal OS window refresh events and key inputs without stalling the image acquisition workflow.

---

## :green_square: Emergency Management & Safety Routines
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
