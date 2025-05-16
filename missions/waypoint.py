"""
Waypoint Missions Module
---------------------
Functions for executing waypoint-based drone missions using pymavlink.
"""

import logging
import time
from threading import Thread
import cv2
from pymavlink import mavutil

from drone.connection import get_vehicle_state  # Corrected import location
from drone.navigation import (
    arm_and_takeoff, set_mode, get_location, get_distance_metres,
    get_location_metres, navigate_to_waypoint, return_to_launch,
    send_ned_velocity
)
from detection.models import load_detection_model, run_detection, process_detection_results
from detection.camera import initialize_camera, capture_frame, close_camera

# Default waypoints for testing (latitude, longitude)
DEFAULT_WAYPOINTS = [
    (35.722952, -120.767658),  # Example waypoint 1
    (35.723101, -120.767592),  # Example waypoint 2
    (35.723072, -120.767421),  # Example waypoint 3
    (35.722925, -120.767489)   # Example waypoint 4
]

# Default relative waypoints for testing (meters North, meters East)
DEFAULT_RELATIVE_WAYPOINTS = [
    (10, 0),    # 10m North
    (10, 10),   # 10m North, 10m East
    (0, 10),    # 10m East
    (0, 0)      # Back to start
]

def mission_waypoint(vehicle, altitude=10, waypoints=None, relative=False):
    """
    Execute a simple waypoint navigation mission.

    Args:
        vehicle: The connected mavlink object
        altitude: Target altitude in meters
        waypoints: List of waypoints to visit (lat, lon) or (dNorth, dEast) if relative
        relative: If True, waypoints are relative to starting position

    Returns:
        True if mission was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Set default waypoints if none provided
        if waypoints is None:
            waypoints = DEFAULT_RELATIVE_WAYPOINTS if relative else DEFAULT_WAYPOINTS

        logging.info(f"Starting waypoint mission with {len(waypoints)} waypoints at {altitude}m altitude")
        logging.info(f"Using {'relative' if relative else 'absolute'} waypoints")

        # First, arm and take off
        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to arm and take off")
            return False

        # Navigate to each waypoint
        original_location = get_location(vehicle)
        if not original_location and relative:
            logging.error("Failed to get current location for relative navigation")
            return_to_launch(vehicle)
            return False

        for i, waypoint in enumerate(waypoints):
            logging.info(f"Navigating to waypoint {i+1}/{len(waypoints)}")

            if relative:
                logging.info(f"Relative waypoint: {waypoint[0]}m North, {waypoint[1]}m East")
                success = navigate_to_waypoint(
                    vehicle, waypoint, altitude, relative=True
                )
            else:
                logging.info(f"Absolute waypoint: Lat {waypoint[0]}, Lon {waypoint[1]}")
                success = navigate_to_waypoint(
                    vehicle, waypoint, altitude, relative=False
                )

            if not success:
                logging.error(f"Failed to navigate to waypoint {i+1}")
                return_to_launch(vehicle)
                return False

            # Hover for 5 seconds at each waypoint
            logging.info(f"Reached waypoint {i+1}. Hovering for 5 seconds")
            time.sleep(5)

        # Return to launch
        logging.info("Mission complete. Returning to launch")
        return_to_launch(vehicle)

        # Wait for landing
        start_time = time.time()
        while time.time() - start_time < 60:  # 1 minute timeout
            # Get latest state
            state = get_vehicle_state(vehicle)
            if state:
                armed = state.get('armed', None)
                if armed is not None and not armed:
                    logging.info("Vehicle has disarmed")
                    break

                altitude = state.get('altitude', None)
                if altitude is not None:
                    logging.info(f"Altitude: {altitude} meters")
                    if altitude < 0.5:  # Close to ground
                        logging.info("Vehicle has landed")
                        break
            time.sleep(1)

        logging.info("Waypoint mission completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during waypoint mission: {str(e)}")
        # Try to return to launch if there was an error
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False

def mission_waypoint_detect(vehicle, altitude=10, model_path=None, waypoints=None, relative=False):
    """
    Execute a waypoint navigation mission with object detection.

    Args:
        vehicle: The connected mavlink object
        altitude: Target altitude in meters
        model_path: Path to the detection model
        waypoints: List of waypoints to visit
        relative: If True, waypoints are relative to starting position

    Returns:
        True if mission was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Set default waypoints if none provided
        if waypoints is None:
            waypoints = DEFAULT_RELATIVE_WAYPOINTS if relative else DEFAULT_WAYPOINTS

        logging.info(f"Starting waypoint detection mission with {len(waypoints)} waypoints")

        # Initialize variables for detection results
        target_detected = False
        detection_center_x = None
        detection_center_y = None
        detection_thread_running = True

        # Load detection model
        model = load_detection_model(model_path)
        if not model:
            logging.error("Failed to load detection model")
            return False

        # Define detection thread function
        def detection_thread():
            nonlocal target_detected, detection_center_x, detection_center_y

            try:
                # Initialize camera
                cap = initialize_camera(0)
                if not cap:
                    logging.error("Failed to initialize camera")
                    return

                # Get camera center coordinates
                cam_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                cam_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                camera_center_x = cam_width / 2
                camera_center_y = cam_height / 2

                logging.info(f"Camera initialized. Resolution: {cam_width}x{cam_height}")

                # Run detection until thread is stopped
                while detection_thread_running:
                    # Capture frame
                    frame = capture_frame(cap)
                    if frame is None:
                        time.sleep(0.1)
                        continue

                    # Run detection on frame
                    results = run_detection(
                        model,
                        source=frame,
                        threshold=0.5,
                        save_results=False
                    )

                    # Process detection results
                    if results:
                        detections = process_detection_results(
                            [next(results)],
                            frame,
                            display=True
                        )

                        # Check if any object was detected
                        if detections:
                            # Use the first detection (highest confidence)
                            detection = detections[0]
                            target_detected = True
                            detection_center_x = detection['center'][0]
                            detection_center_y = detection['center'][1]

                            logging.info(f"Target detected at ({detection_center_x}, {detection_center_y})")
                        else:
                            target_detected = False
                            detection_center_x = None
                            detection_center_y = None

                    # Sleep briefly to reduce CPU usage
                    time.sleep(0.1)

                # Clean up resources
                close_camera(cap)

            except Exception as e:
                logging.error(f"Error in detection thread: {str(e)}")

        # Start detection thread
        det_thread = Thread(target=detection_thread)
        det_thread.daemon = True
        det_thread.start()

        # Wait for detection thread to initialize
        time.sleep(2)

        # First, arm and take off
        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to arm and take off")
            detection_thread_running = False
            return False

        # Navigate to each waypoint
        original_location = get_location(vehicle)
        if not original_location and relative:
            logging.error("Failed to get current location for relative navigation")
            detection_thread_running = False
            return_to_launch(vehicle)
            return False

        for i, waypoint in enumerate(waypoints):
            logging.info(f"Navigating to waypoint {i+1}/{len(waypoints)}")

            if relative:
                success = navigate_to_waypoint(
                    vehicle, waypoint, altitude, relative=True
                )
            else:
                success = navigate_to_waypoint(
                    vehicle, waypoint, altitude, relative=False
                )

            if not success:
                logging.error(f"Failed to navigate to waypoint {i+1}")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Hover for 5 seconds at each waypoint to run detection
            logging.info(f"Reached waypoint {i+1}. Hovering for 5 seconds")

            # Check for detections at this waypoint
            detection_start = time.time()
            while time.time() - detection_start < 5:
                if target_detected:
                    logging.info("Target detected! Processing...")

                    # Handle the detection here
                    # For now, just log it
                    logging.info(f"Detected target at coordinates: ({detection_center_x}, {detection_center_y})")

                    # You could add code here to align to the target, drop a package, etc.

                    # For demo purposes, hover a bit longer when target is found
                    time.sleep(2)
                    break

                time.sleep(0.5)

        # Return to launch
        logging.info("Mission complete. Returning to launch")
        return_to_launch(vehicle)

        # Wait for landing
        start_time = time.time()
        while time.time() - start_time < 60:  # 1 minute timeout
            # Get latest state
            state = get_vehicle_state(vehicle)
            if state:
                armed = state.get('armed', None)
                if armed is not None and not armed:
                    logging.info("Vehicle has disarmed")
                    break

                altitude = state.get('altitude', None)
                if altitude is not None:
                    logging.info(f"Altitude: {altitude} meters")
                    if altitude < 0.5:  # Close to ground
                        logging.info("Vehicle has landed")
                        break
            time.sleep(1)

        # Stop detection thread
        detection_thread_running = False
        det_thread.join(timeout=2)  # Wait for thread to finish

        logging.info("Waypoint detection mission completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during waypoint detection mission: {str(e)}")
        # Stop detection thread
        detection_thread_running = False
        # Try to return to launch if there was an error
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False

def follow_mission_file(vehicle, mission_file):
    """
    Load and execute a mission from a file.

    Args:
        vehicle: The connected mavlink object
        mission_file: Path to mission file

    Returns:
        True if mission was loaded and started successfully, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Loading mission from file: {mission_file}")

        # TODO: Implement mission file loading
        # This would typically involve parsing a mission file format (e.g., .waypoints, .mission)
        # and uploading the waypoints to the vehicle using MAVLink mission protocol

        # For now, just log that this feature is not implemented
        logging.warning("Mission file loading not implemented yet")
        return False
    except Exception as e:
        logging.error(f"Error loading mission file: {str(e)}")
        return False
