"""
Package Delivery Missions Module
-----------------------------
Functions for executing package delivery missions using pymavlink.
"""

import logging
import time
import math
from threading import Thread
import cv2
from pymavlink import mavutil

from drone.connection import get_vehicle_state  # Corrected import location
from drone.navigation import (
    arm_and_takeoff, get_location, navigate_to_waypoint,
    return_to_launch, send_ned_velocity
)
from drone.servo import operate_package_release, operate_claw
from detection.models import load_detection_model, run_detection, process_detection_results
from detection.camera import initialize_camera, capture_frame, close_camera
from detection.gcp import detect_gcp_markers

def align_to_target(vehicle, target_x, target_y, camera_x, camera_y, timeout=30):
    """
    Align the drone to a detected target.

    Args:
        vehicle: The connected mavlink object
        target_x: Target X coordinate in the image
        target_y: Target Y coordinate in the image
        camera_x: Camera center X coordinate
        camera_y: Camera center Y coordinate
        timeout: Maximum alignment time in seconds

    Returns:
        True if alignment was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Aligning drone to target")

        # Calculate offset
        offset_x = target_x - camera_x
        offset_y = target_y - camera_y

        # Define alignment threshold (pixels)
        threshold = 20

        # Start alignment timer
        start_time = time.time()

        while (abs(offset_x) > threshold or abs(offset_y) > threshold) and time.time() - start_time < timeout:
            logging.info(f"Target offset: X={offset_x}, Y={offset_y}")

            # Calculate velocity based on offset
            # Note: X offset maps to Y velocity, Y offset maps to X velocity in camera frame
            # Scale velocity based on how far we are from the target (proportional control)
            velocity_scale = 0.2  # m/s maximum speed
            velocity_x = velocity_scale * (offset_y / max(abs(offset_y), 100)) if abs(offset_y) > threshold else 0
            velocity_y = velocity_scale * (offset_x / max(abs(offset_x), 100)) if abs(offset_x) > threshold else 0

            logging.info(f"Adjustment velocity: X={velocity_x}, Y={velocity_y}")

            # Send velocity command
            send_ned_velocity(vehicle, velocity_x, velocity_y, 0, 1)

            # Wait for drone to move
            time.sleep(1)

            # Recalculate offset (this would come from detection thread in practice)
            # This is a stub - in a real implementation, you'd get updated coordinates
            # from the detection system
            offset_x = offset_x * 0.7  # Simulate getting closer
            offset_y = offset_y * 0.7  # Simulate getting closer

        if time.time() - start_time >= timeout:
            logging.warning("Alignment timed out")
            return False

        logging.info("Drone aligned to target")
        return True
    except Exception as e:
        logging.error(f"Error during target alignment: {str(e)}")
        return False

def lower_to_delivery_height(vehicle, target_height, speed=0.5):
    """
    Lower the drone to a delivery height.

    Args:
        vehicle: The connected mavlink object
        target_height: Target height in meters
        speed: Descent speed in m/s

    Returns:
        True if lowering was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Get current height
        state = get_vehicle_state(vehicle)
        if not state or state['altitude'] is None:
            logging.error("Could not get current altitude")
            return False

        current_height = state['altitude']

        logging.info(f"Lowering drone from {current_height}m to {target_height}m")

        # Calculate descent distance
        descent_distance = current_height - target_height
        if descent_distance <= 0:
            logging.warning("Already at or below target height")
            return True

        # Calculate descent time
        descent_time = descent_distance / speed

        # Send descent velocity command (positive Z is down)
        send_ned_velocity(vehicle, 0, 0, speed, descent_time)

        # Wait for drone to reach target height
        timeout = descent_time + 5  # Add safety margin
        start_time = time.time()

        while time.time() - start_time < timeout:
            state = get_vehicle_state(vehicle)
            if state and state['altitude'] is not None:
                current_alt = state['altitude']
                logging.info(f"Current altitude: {current_alt}m")

                if current_alt <= target_height + 0.5:  # Allow 0.5m tolerance
                    logging.info(f"Reached delivery height: {current_alt}m")
                    return True

            time.sleep(1)

        logging.warning("Lowering timed out")
        return False
    except Exception as e:
        logging.error(f"Error during lowering: {str(e)}")
        return False

def raise_to_safe_height(vehicle, target_height, speed=0.5):
    """
    Raise the drone to a safe height after delivery.

    Args:
        vehicle: The connected mavlink object
        target_height: Target height in meters
        speed: Ascent speed in m/s

    Returns:
        True if raising was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Get current height
        state = get_vehicle_state(vehicle)
        if not state or state['altitude'] is None:
            logging.error("Could not get current altitude")
            return False

        current_height = state['altitude']

        logging.info(f"Raising drone from {current_height}m to {target_height}m")

        # Calculate ascent distance
        ascent_distance = target_height - current_height
        if ascent_distance <= 0:
            logging.warning("Already at or above target height")
            return True

        # Calculate ascent time
        ascent_time = ascent_distance / speed

        # Send ascent velocity command (negative Z is up)
        send_ned_velocity(vehicle, 0, 0, -speed, ascent_time)

        # Wait for drone to reach target height
        timeout = ascent_time + 5  # Add safety margin
        start_time = time.time()

        while time.time() - start_time < timeout:
            state = get_vehicle_state(vehicle)
            if state and state['altitude'] is not None:
                current_alt = state['altitude']
                logging.info(f"Current altitude: {current_alt}m")

                if current_alt >= target_height - 0.5:  # Allow 0.5m tolerance
                    logging.info(f"Reached safe height: {current_alt}m")
                    return True

            time.sleep(1)

        logging.warning("Raising timed out")
        return False
    except Exception as e:
        logging.error(f"Error during raising: {str(e)}")
        return False

def mission_package_delivery(vehicle, altitude=10, model_path=None, delivery_height=2):
    """
    Execute a package delivery mission with landing at target.

    Args:
        vehicle: The connected mavlink object
        altitude: Cruising altitude in meters
        model_path: Path to the detection model
        delivery_height: Height to lower to for delivery

    Returns:
        True if mission was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Starting package delivery mission")

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

        # Start a search pattern (simple square for now)
        waypoints = [
            (10, 0),    # 10m North
            (10, 10),   # 10m North, 10m East
            (0, 10),    # 10m East
            (0, 0)      # Back to start
        ]

        # Search for target
        target_found = False

        for i, waypoint in enumerate(waypoints):
            logging.info(f"Navigating to search point {i+1}/{len(waypoints)}")

            success = navigate_to_waypoint(
                vehicle, waypoint, altitude, relative=True
            )

            if not success:
                logging.error(f"Failed to navigate to search point {i+1}")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Hover for 5 seconds at each waypoint to run detection
            logging.info(f"Reached search point {i+1}. Scanning for target")

            # Check for detections at this waypoint
            detection_start = time.time()
            while time.time() - detection_start < 5:
                if target_detected:
                    logging.info("Target detected! Preparing for delivery")
                    target_found = True
                    break

                time.sleep(0.5)

            if target_found:
                break

        # If target was found, align and deliver package
        if target_found and detection_center_x is not None and detection_center_y is not None:
            logging.info("Target found. Proceeding with delivery")

            # Align to target
            camera_center_x = 640 / 2  # Assuming 640x480 resolution
            camera_center_y = 480 / 2

            if not align_to_target(
                vehicle, detection_center_x, detection_center_y,
                camera_center_x, camera_center_y
            ):
                logging.error("Failed to align to target")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Lower to delivery height
            if not lower_to_delivery_height(vehicle, delivery_height):
                logging.error("Failed to lower to delivery height")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Deploy package
            logging.info("Deploying package")
            operate_claw(vehicle)
            time.sleep(2)

            # Raise back to safe height
            if not raise_to_safe_height(vehicle, altitude):
                logging.error("Failed to raise to safe height")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False
        else:
            logging.warning("Target not found during search pattern")

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

        logging.info("Package delivery mission completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during package delivery mission: {str(e)}")
        # Stop detection thread
        detection_thread_running = False
        # Try to return to launch if there was an error
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False

def mission_package_drop(vehicle, altitude=10, model_path=None, drop_altitude=8):
    """
    Execute a package drop mission without lowering.

    Args:
        vehicle: The connected mavlink object
        altitude: Cruising altitude in meters
        model_path: Path to the detection model
        drop_altitude: Altitude at which to drop the package

    Returns:
        True if mission was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Starting package drop mission")

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

        # Start a search pattern (simple square for now)
        waypoints = [
            (10, 0),    # 10m North
            (10, 10),   # 10m North, 10m East
            (0, 10),    # 10m East
            (0, 0)      # Back to start
        ]

        # Search for target
        target_found = False

        for i, waypoint in enumerate(waypoints):
            logging.info(f"Navigating to search point {i+1}/{len(waypoints)}")

            success = navigate_to_waypoint(
                vehicle, waypoint, altitude, relative=True
            )

            if not success:
                logging.error(f"Failed to navigate to search point {i+1}")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Hover for 5 seconds at each waypoint to run detection
            logging.info(f"Reached search point {i+1}. Scanning for target")

            # Check for detections at this waypoint
            detection_start = time.time()
            while time.time() - detection_start < 5:
                if target_detected:
                    logging.info("Target detected! Preparing for drop")
                    target_found = True
                    break

                time.sleep(0.5)

            if target_found:
                break

        # If target was found, align and drop package
        if target_found and detection_center_x is not None and detection_center_y is not None:
            logging.info("Target found. Proceeding with package drop")

            # Align to target
            camera_center_x = 640 / 2  # Assuming 640x480 resolution
            camera_center_y = 480 / 2

            if not align_to_target(
                vehicle, detection_center_x, detection_center_y,
                camera_center_x, camera_center_y
            ):
                logging.error("Failed to align to target")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Drop package
            logging.info("Dropping package")
            operate_package_release(vehicle)
            time.sleep(2)
        else:
            logging.warning("Target not found during search pattern")

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

        logging.info("Package drop mission completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during package drop mission: {str(e)}")
        # Stop detection thread
        detection_thread_running = False
        # Try to return to launch if there was an error
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False

def mission_target_localize(vehicle, altitude=10):
    """
    Execute a mission to locate ground control points.

    Args:
        vehicle: The connected mavlink object
        altitude: Cruising altitude in meters

    Returns:
        True if mission was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Starting target localization mission")

        # Initialize variables for GCP detection
        target_found = False
        target_type = None
        target_position = None
        detection_thread_running = True

        # Define GCP detection thread function
        def gcp_detection_thread():
            nonlocal target_found, target_type, target_position

            try:
                # Initialize camera
                cap = initialize_camera(0)
                if not cap:
                    logging.error("Failed to initialize camera")
                    return

                logging.info("Camera initialized for GCP detection")

                # Run detection until thread is stopped
                while detection_thread_running:
                    # Capture frame
                    frame = capture_frame(cap)
                    if frame is None:
                        time.sleep(0.1)
                        continue

                    # Detect GCP markers
                    results = detect_gcp_markers(frame, save_debug=True)

                    # Check for X-patterns (highest priority)
                    if len(results['x_patterns']) > 0:
                        x, y, w, h, conf = results['x_patterns'][0]
                        target_found = True
                        target_type = 'x_pattern'
                        target_position = (x + w/2, y + h/2)
                        logging.info(f"X-Pattern detected at {target_position} with confidence {conf}")

                    # Check for triangles in squares
                    elif len(results['tri_in_squares']) > 0:
                        x, y, w, h, count = results['tri_in_squares'][0]
                        target_found = True
                        target_type = 'triangles_in_square'
                        target_position = (x + w/2, y + h/2)
                        logging.info(f"Triangles in square detected at {target_position} with {count} triangles")

                    # Check for squares
                    elif len(results['squares']) > 0:
                        x, y, w, h, _ = results['squares'][0]
                        target_found = True
                        target_type = 'square'
                        target_position = (x + w/2, y + h/2)
                        logging.info(f"Square detected at {target_position}")

                    # Check for triangles
                    elif len(results['triangles']) > 0:
                        x, y, w, h, _ = results['triangles'][0]
                        target_found = True
                        target_type = 'triangle'
                        target_position = (x + w/2, y + h/2)
                        logging.info(f"Triangle detected at {target_position}")

                    else:
                        target_found = False
                        target_type = None
                        target_position = None

                    # Display the frame with detections
                    cv2.imshow("GCP Detection", results['display_frame'])
                    cv2.waitKey(1)

                    # Sleep briefly to reduce CPU usage
                    time.sleep(0.1)

                # Clean up resources
                close_camera(cap)
                cv2.destroyAllWindows()

            except Exception as e:
                logging.error(f"Error in GCP detection thread: {str(e)}")

        # Start detection thread
        det_thread = Thread(target=gcp_detection_thread)
        det_thread.daemon = True
        det_thread.start()

        # Wait for detection thread to initialize
        time.sleep(2)

        # First, arm and take off
        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to arm and take off")
            detection_thread_running = False
            return False

        # Start a search pattern
        waypoints = [
            (10, 0),    # 10m North
            (10, 10),   # 10m North, 10m East
            (0, 10),    # 10m East
            (0, 0)      # Back to start
        ]

        # Search for GCP targets
        for i, waypoint in enumerate(waypoints):
            logging.info(f"Navigating to search point {i+1}/{len(waypoints)}")

            success = navigate_to_waypoint(
                vehicle, waypoint, altitude, relative=True
            )

            if not success:
                logging.error(f"Failed to navigate to search point {i+1}")
                detection_thread_running = False
                return_to_launch(vehicle)
                return False

            # Hover for 10 seconds at each waypoint to run detection
            logging.info(f"Reached search point {i+1}. Scanning for GCP targets")

            # Check for detections at this waypoint
            start_time = time.time()
            while time.time() - start_time < 10:
                if target_found:
                    logging.info(f"GCP target found: {target_type}")

                    # Record location if target is found
                    location = get_location(vehicle)
                    if location:
                        logging.info(f"GCP target location: Lat {location[0]}, Lon {location[1]}, Alt {location[2]}")
                        logging.info(f"Target pixel position: {target_position}")

                    # Hover for an additional 5 seconds to gather more data
                    time.sleep(5)
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

        logging.info("Target localization mission completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during target localization mission: {str(e)}")
        # Stop detection thread
        detection_thread_running = False
        # Try to return to launch if there was an error
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False
