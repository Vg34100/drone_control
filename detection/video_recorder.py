# detection/video_recorder.py - NEW MODULE
"""
Video Recording Module
---------------------
Functions for recording camera footage during missions.
"""

import cv2
import time
import logging
import os
import threading
from datetime import datetime
from typing import Optional

class VideoRecorder:
    """Video recorder class for mission recording"""

    def __init__(self, output_dir="recordings", fps=30.0, codec='XVID'):
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.recording = False
        self.writer = None
        self.cap = None
        self.record_thread = None
        self.current_filename = None

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def start_recording(self, camera_id=0, resolution=(640, 480), mission_name="mission"):
        """
        Start recording video from camera.

        Args:
            camera_id: Camera ID to record from
            resolution: Video resolution (width, height)
            mission_name: Name of mission for filename

        Returns:
            True if recording started successfully
        """
        if self.recording:
            logging.warning("Recording already in progress")
            return False

        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera {camera_id}")
                return False

            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture test frame")
                self.cap.release()
                return False

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_filename = f"{mission_name}_{timestamp}.avi"
            full_path = os.path.join(self.output_dir, self.current_filename)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.writer = cv2.VideoWriter(full_path, fourcc, self.fps, (actual_width, actual_height))

            if not self.writer.isOpened():
                logging.error("Failed to initialize video writer")
                self.cap.release()
                return False

            # Start recording thread
            self.recording = True
            self.record_thread = threading.Thread(target=self._record_loop)
            self.record_thread.daemon = True
            self.record_thread.start()

            logging.info(f"Recording started: {full_path}")
            logging.info(f"Resolution: {actual_width}x{actual_height} @ {self.fps} FPS")

            return True

        except Exception as e:
            logging.error(f"Error starting recording: {str(e)}")
            self.cleanup()
            return False

    def _record_loop(self):
        """Main recording loop (runs in separate thread)"""
        frame_count = 0
        start_time = time.time()

        try:
            while self.recording and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    logging.warning("Failed to capture frame, stopping recording")
                    break

                # Write frame to video file
                if self.writer:
                    self.writer.write(frame)
                    frame_count += 1

                # Brief sleep to prevent 100% CPU usage
                time.sleep(0.001)

        except Exception as e:
            logging.error(f"Error in recording loop: {str(e)}")

        # Calculate recording stats
        duration = time.time() - start_time
        actual_fps = frame_count / duration if duration > 0 else 0

        logging.info(f"Recording stopped: {frame_count} frames in {duration:.1f}s ({actual_fps:.1f} FPS)")

    def stop_recording(self):
        """Stop recording and save file"""
        if not self.recording:
            logging.warning("No recording in progress")
            return False

        try:
            self.recording = False

            # Wait for recording thread to finish
            if self.record_thread:
                self.record_thread.join(timeout=5)

            self.cleanup()

            if self.current_filename:
                full_path = os.path.join(self.output_dir, self.current_filename)
                if os.path.exists(full_path):
                    file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                    logging.info(f"Recording saved: {full_path} ({file_size:.1f} MB)")
                    return True
                else:
                    logging.error("Recording file not found after stopping")
                    return False

            return True

        except Exception as e:
            logging.error(f"Error stopping recording: {str(e)}")
            return False

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.writer:
                self.writer.release()
                self.writer = None

            if self.cap:
                self.cap.release()
                self.cap = None

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def is_recording(self):
        """Check if currently recording"""
        return self.recording

    def get_current_filename(self):
        """Get current recording filename"""
        return self.current_filename

def create_video_recorder(output_dir="recordings", fps=30.0, codec='XVID'):
    """
    Factory function to create a video recorder.

    Args:
        output_dir: Directory to save recordings
        fps: Frames per second
        codec: Video codec ('XVID', 'MJPG', 'H264')

    Returns:
        VideoRecorder instance
    """
    return VideoRecorder(output_dir, fps, codec)

def test_video_recording(camera_id=0, duration=10, output_dir="test_recordings"):
    """
    Test video recording functionality.

    Args:
        camera_id: Camera ID to use
        duration: Recording duration in seconds
        output_dir: Output directory for test recording

    Returns:
        True if test successful
    """
    try:
        logging.info(f"Testing video recording for {duration} seconds")

        recorder = create_video_recorder(output_dir)

        # Start recording
        if not recorder.start_recording(camera_id, mission_name="test"):
            logging.error("Failed to start test recording")
            return False

        # Record for specified duration
        time.sleep(duration)

        # Stop recording
        if not recorder.stop_recording():
            logging.error("Failed to stop test recording")
            return False

        logging.info("Video recording test completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error during video recording test: {str(e)}")
        return False
