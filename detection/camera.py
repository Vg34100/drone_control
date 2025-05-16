"""
Camera Module
------------
Functions for managing camera operations, including setup and testing.
"""

import cv2
import time
import logging
import os
import numpy as np

def initialize_camera(camera_id=0, resolution=(640, 480)):
    """
    Initialize the camera.

    Args:
        camera_id: Camera ID (default: 0 for primary camera)
        resolution: Tuple of (width, height) for desired resolution

    Returns:
        Initialized camera object or None if initialization failed
    """
    try:
        cap = cv2.VideoCapture(camera_id)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Check if camera opened successfully
        if not cap.isOpened():
            logging.error(f"Failed to open camera {camera_id}")
            return None

        # Read a test frame to confirm camera is working
        ret, _ = cap.read()
        if not ret:
            logging.error(f"Failed to read from camera {camera_id}")
            cap.release()
            return None

        logging.info(f"Camera {camera_id} initialized at resolution {resolution}")
        return cap
    except Exception as e:
        logging.error(f"Error initializing camera: {str(e)}")
        return None

def capture_frame(cap):
    """
    Capture a single frame from the camera.

    Args:
        cap: The initialized camera object

    Returns:
        Captured frame or None if capture failed
    """
    if not cap or not cap.isOpened():
        logging.error("Invalid camera object")
        return None

    try:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            return None

        return frame
    except Exception as e:
        logging.error(f"Error capturing frame: {str(e)}")
        return None

def save_frame(frame, output_dir="debug_frames", filename=None):
    """
    Save a frame to disk.

    Args:
        frame: The frame to save
        output_dir: Directory to save the frame
        filename: Filename (if None, use timestamp)

    Returns:
        Path to saved file or None if save failed
    """
    if frame is None:
        logging.error("Cannot save None frame")
        return None

    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate filename based on timestamp if not provided
        if filename is None:
            filename = f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Full path to save the frame
        file_path = os.path.join(output_dir, filename)

        # Save the frame
        cv2.imwrite(file_path, frame)
        logging.info(f"Frame saved to {file_path}")

        return file_path
    except Exception as e:
        logging.error(f"Error saving frame: {str(e)}")
        return None

def test_camera_feed(camera_id=0, duration=10, show_preview=True):
    """
    Test the camera by capturing frames for a specified duration.

    Args:
        camera_id: Camera ID
        duration: Test duration in seconds
        show_preview: Whether to show preview window

    Returns:
        True if test was successful, False otherwise
    """
    try:
        cap = initialize_camera(camera_id)
        if not cap:
            return False

        start_time = time.time()
        frame_count = 0

        logging.info(f"Starting camera test for {duration} seconds")

        # Create debug directory if showing preview
        debug_dir = "debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        while time.time() - start_time < duration:
            frame = capture_frame(cap)
            if frame is None:
                break

            frame_count += 1

            # Add frame counter and timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Frame: {frame_count} Time: {timestamp}",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame if preview is enabled
            if show_preview:
                cv2.imshow('Camera Test', frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save every 30th frame
            if frame_count % 30 == 0:
                save_frame(frame, debug_dir, f"test_{frame_count}.jpg")

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Clean up
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        logging.info(f"Camera test completed. Captured {frame_count} frames in {elapsed_time:.1f} seconds ({fps:.1f} FPS)")
        return frame_count > 0
    except Exception as e:
        logging.error(f"Error during camera test: {str(e)}")
        return False

def close_camera(cap):
    """
    Safely release the camera resource.

    Args:
        cap: The camera object to close
    """
    if cap and cap.isOpened():
        try:
            cap.release()
            logging.info("Camera released")
        except Exception as e:
            logging.error(f"Error releasing camera: {str(e)}")

def adjust_camera_settings(cap, brightness=None, contrast=None, saturation=None,
                         exposure=None, auto_exposure=None):
    """
    Adjust camera settings like brightness, contrast, etc.

    Args:
        cap: The initialized camera object
        brightness: Brightness value (typically 0-100)
        contrast: Contrast value (typically 0-100)
        saturation: Saturation value (typically 0-100)
        exposure: Exposure value
        auto_exposure: Auto exposure mode (0: manual, 1: auto)

    Returns:
        True if settings were adjusted successfully, False otherwise
    """
    if not cap or not cap.isOpened():
        logging.error("Invalid camera object")
        return False

    try:
        # Set brightness if specified
        if brightness is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            logging.info(f"Camera brightness set to {brightness}")

        # Set contrast if specified
        if contrast is not None:
            cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            logging.info(f"Camera contrast set to {contrast}")

        # Set saturation if specified
        if saturation is not None:
            cap.set(cv2.CAP_PROP_SATURATION, saturation)
            logging.info(f"Camera saturation set to {saturation}")

        # Set auto exposure mode if specified
        if auto_exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
            logging.info(f"Camera auto exposure set to {auto_exposure}")

        # Set exposure if specified and auto exposure is off
        if exposure is not None and (auto_exposure is None or auto_exposure == 0):
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            logging.info(f"Camera exposure set to {exposure}")

        return True
    except Exception as e:
        logging.error(f"Error adjusting camera settings: {str(e)}")
        return False

def get_camera_properties(cap):
    """
    Get current camera properties.

    Args:
        cap: The initialized camera object

    Returns:
        Dictionary of camera properties or None if failed
    """
    if not cap or not cap.isOpened():
        logging.error("Invalid camera object")
        return None

    try:
        properties = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        }

        logging.info("Retrieved camera properties")
        return properties
    except Exception as e:
        logging.error(f"Error getting camera properties: {str(e)}")
        return None

def list_available_cameras(max_cameras=10):
    """
    List available cameras by trying to open each one.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of available camera IDs
    """
    available_cameras = []

    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except:
            pass

    if available_cameras:
        logging.info(f"Found {len(available_cameras)} available cameras: {available_cameras}")
    else:
        logging.warning("No available cameras found")

    return available_cameras
