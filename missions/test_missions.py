"""
Test Missions Module
-----------------
Functions for testing drone components and functionality using pymavlink.
"""

import logging
import time
import cv2

from drone.connection import get_vehicle_state, print_vehicle_state
from drone.navigation import (
    arm_vehicle, disarm_vehicle, set_mode, arm_and_takeoff,
    return_to_launch, check_if_armed, test_motors
)
from detection.camera import test_camera_feed
from detection.models import load_detection_model, test_detection_model

def test_connection(vehicle):
    """
    Test the connection to the drone by checking its state.

    Args:
        vehicle: The connected mavlink object

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Testing vehicle connection")

        # Get and print vehicle state
        state = get_vehicle_state(vehicle)
        if not state:
            logging.error("Failed to get vehicle state")
            return False

        print_vehicle_state(vehicle)

        # Request a heartbeat to verify communication
        vehicle.mav.heartbeat_send(
            6,                  # Type: MAV_TYPE_GCS
            8,                  # Autopilot: MAV_AUTOPILOT_INVALID
            0,                  # Base mode: None
            0,                  # Custom mode: None
            0                   # System status: None
        )

        # Wait for heartbeat response
        msg = vehicle.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
        if not msg:
            logging.error("No heartbeat received from vehicle")
            return False

        logging.info(f"Received heartbeat from system {vehicle.target_system}, component {vehicle.target_component}")

        # Success if we've gotten this far
        logging.info("Connection test completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during connection test: {str(e)}")
        return False

def test_arm(vehicle, duration=3):
    """
    Test arming and disarming the vehicle.

    Args:
        vehicle: The connected mavlink object
        duration: Duration to keep armed in seconds

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Testing arm functionality")

        # Check current arm state
        armed = check_if_armed(vehicle)
        if armed:
            logging.warning("Vehicle is already armed. Disarming first...")
            if not disarm_vehicle(vehicle):
                logging.error("Failed to disarm vehicle")
                return False

        # Set to GUIDED mode
        if not set_mode(vehicle, "GUIDED"):
            logging.error("Failed to set GUIDED mode")
            return False

        # Arm the vehicle
        logging.info("Arming vehicle")
        if not arm_vehicle(vehicle, force=False):
            logging.error("Failed to arm vehicle")
            return False

        # Verify armed state
        armed = check_if_armed(vehicle)
        if not armed:
            logging.error("Vehicle did not arm successfully")
            return False

        logging.info(f"Vehicle armed. Waiting for {duration} seconds...")
        time.sleep(duration)

        # Disarm the vehicle
        logging.info("Disarming vehicle")
        if not disarm_vehicle(vehicle):
            logging.error("Failed to disarm vehicle")
            return False

        # Verify disarmed state
        armed = check_if_armed(vehicle)
        if armed:
            logging.error("Vehicle did not disarm successfully")
            return False

        logging.info("Arm test completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during arm test: {str(e)}")

        # Try to disarm if there was an error
        try:
            disarm_vehicle(vehicle)
        except:
            pass

        return False

def test_takeoff(vehicle, altitude=3):
    """
    Test the drone takeoff and landing process.

    Args:
        vehicle: The connected mavlink object
        altitude: Target altitude in meters

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Testing takeoff to {altitude} meters")

        # First, arm and takeoff
        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to arm and takeoff")
            return False

        # Hover for 10 seconds
        logging.info("Hovering for 10 seconds")
        for i in range(10):
            logging.info(f"Hovering... {i+1}/10 seconds")
            print_vehicle_state(vehicle)
            time.sleep(1)

        # Return to launch
        logging.info("Testing return to launch")
        if not return_to_launch(vehicle):
            logging.error("Failed to return to launch")
            return False

        # Wait for landing and disarm
        logging.info("Waiting for landing")
        start_time = time.time()
        while check_if_armed(vehicle) and time.time() - start_time < 60:
            state = get_vehicle_state(vehicle)
            if state and 'altitude' in state:
                logging.info(f"Altitude: {state['altitude']} meters")
            time.sleep(1)

        if check_if_armed(vehicle):
            logging.warning("Vehicle still armed after RTL - trying manual disarm")
            disarm_vehicle(vehicle)

        logging.info("Takeoff test completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during takeoff test: {str(e)}")

        # Try to return to launch if there was an error
        try:
            return_to_launch(vehicle)
            time.sleep(5)
            disarm_vehicle(vehicle)
        except:
            pass

        return False

def test_camera(camera_id=0, duration=10):
    """
    Test the camera feed.

    Args:
        camera_id: Camera ID to use
        duration: Test duration in seconds

    Returns:
        True if test was successful, False otherwise
    """
    try:
        logging.info(f"Testing camera {camera_id} for {duration} seconds")
        return test_camera_feed(camera_id, duration)
    except Exception as e:
        logging.error(f"Error during camera test: {str(e)}")
        return False

def test_detection(model_path, test_source="0", duration=10):
    """
    Test the object detection model.

    Args:
        model_path: Path to the detection model
        test_source: Source for testing (0 for webcam, or path to image/video)
        duration: Test duration in seconds

    Returns:
        True if test was successful, False otherwise
    """
    try:
        logging.info(f"Testing detection model {model_path}")

        # Load the model
        model = load_detection_model(model_path)
        if not model:
            logging.error("Failed to load detection model")
            return False

        # Test the model
        return test_detection_model(model, test_source, duration=duration)
    except Exception as e:
        logging.error(f"Error during detection test: {str(e)}")
        return False

def test_motor(vehicle, throttle_percentage=15, duration_per_motor=1):
    """
    Test each motor individually.

    Args:
        vehicle: The connected mavlink object
        throttle_percentage: Throttle percentage (0-100)
        duration_per_motor: Duration to run each motor in seconds

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Testing motors at {throttle_percentage}% throttle for {duration_per_motor}s each")

        # Check if vehicle is in the air
        state = get_vehicle_state(vehicle)
        if state and state['armed']:
            is_flying = state['altitude'] > 0.5 if state['altitude'] is not None else True
            if is_flying:
                logging.error("Cannot run motor test while vehicle is armed or flying")
                return False

        # Run motor test
        if not test_motors(vehicle, throttle_percentage, duration_per_motor):
            logging.error("Motor test failed")
            return False

        logging.info("Motor test completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during motor test: {str(e)}")
        return False

def test_all(vehicle, model_path, altitude=3, camera_id=0):
    """
    Run all tests sequentially.

    Args:
        vehicle: The connected mavlink object
        model_path: Path to the detection model
        altitude: Target altitude in meters
        camera_id: Camera ID to use

    Returns:
        Dictionary with test results
    """
    results = {}

    # Test connection
    logging.info("=== STARTING CONNECTION TEST ===")
    results['connection'] = test_connection(vehicle)

    # Test camera
    logging.info("=== STARTING CAMERA TEST ===")
    results['camera'] = test_camera(camera_id)

    # Test detection
    logging.info("=== STARTING DETECTION TEST ===")
    results['detection'] = test_detection(model_path)

    # Test arm
    logging.info("=== STARTING ARM TEST ===")
    results['arm'] = test_arm(vehicle)

    # Test motors (CAUTION: only if safe)
    logging.info("=== SKIPPING MOTOR TEST (Run individually if needed) ===")
    results['motor'] = False

    # Test takeoff (last since it involves actual flight)
    logging.info("=== STARTING TAKEOFF TEST ===")
    results['takeoff'] = test_takeoff(vehicle, altitude)

    # Print summary
    logging.info("=== TEST SUMMARY ===")
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        logging.info(f"Test '{test}': {status}")

    # Overall result
    results['all_passed'] = all(results.values())

    return results
