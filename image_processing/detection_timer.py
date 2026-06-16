class DetectionTimer:
    def __init__(self, target_ms, grace_ms=300.0):
        """
        [Arguments]
        target_ms: Target duration to maintain detection in milliseconds (Required).
        grace_ms:  Grace period to allow momentary loss in milliseconds (Default: 300.0ms).
        """
        self.TARGET_MS = target_ms
        self.GRACE_MS = grace_ms
        
        self.start_time = None
        self.lost_time = None
        self.is_reached = False

    def update(self, is_detected, current_msec):
        """
        Updates the internal timer state every frame.
        [Arguments]
        is_detected:  Boolean flag whether the target is detected in the current frame.
        current_msec: Current timestamp of the video stream in milliseconds.
        [Returns]
        Returns True continuously once the target has been tracked for the target_ms duration.
        """
        if self.is_reached:
            return True

        if is_detected:
            self.lost_time = None
            if self.start_time is None:
                self.start_time = current_msec  # Start counting
            
            # Check if the accumulated time has reached the target duration
            if (current_msec - self.start_time) >= self.TARGET_MS:
                self.is_reached = True
        else:
            if self.start_time is not None:
                if self.lost_time is None:
                    self.lost_time = current_msec  # Record the timestamp when the target was lost
                
                # Reset the tracking timer if the loss duration exceeds the grace period
                if (current_msec - self.lost_time) > self.GRACE_MS:
                    self.start_time = None
                    self.lost_time = None
                    
        return self.is_reached
 