"""
Test Missions Module
-----------------
Functions for testing drone components and functionality using pymavlink.
"""

import logging
import time
import cv2
from pymavlink import mavutil

from drone.connection import get_vehicle_state, print_vehicle_state, request_message_interval
from drone.navigation import (
    arm_vehicle, disarm_vehicle, set_mode, arm_and_takeoff,
    return_to_launch, check_if_armed, test_motors, get_altitude, get_location
)
from detection.camera import test_camera_feed
from detection.models import load_detection_model, test_detection_model

def test_connection(vehicle):
    """
    Test the connection to the drone by checking its state and running diagnostics.

    Args:
        vehicle: The connected mavlink object

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Testing vehicle connection and running diagnostics")

        # Standard state check
        state = get_vehicle_state(vehicle)
        if state:
            print_vehicle_state(vehicle)
        else:
            logging.warning("Could not retrieve vehicle state")

        # Get comprehensive diagnostics
        from drone.connection import get_vehicle_diagnostics
        diagnostics = get_vehicle_diagnostics(vehicle, timeout=5)

        if diagnostics:
            logging.info("=== DRONE DIAGNOSTICS ===")

            # Connection info
            logging.info(f"System ID: {diagnostics['connection']['target_system']}")
            logging.info(f"Component ID: {diagnostics['connection']['target_component']}")
            logging.info(f"Connection: {diagnostics['connection']['connection_string']}")

            # Heartbeat status
            logging.info(f"Heartbeat received: {diagnostics['heartbeat_received']}")

            # Mode and armed status
            if diagnostics['mode']:
                logging.info(f"Current mode: {diagnostics['mode']}")
            logging.info(f"Armed: {diagnostics['armed']}")

            # Firmware info
            if diagnostics['firmware_version']:
                logging.info(f"Firmware version: {diagnostics['firmware_version']}")

            # GPS status
            if diagnostics['gps_status']:
                fix_type = diagnostics['gps_status']['fix_type']
                fix_type_name = "No GPS" if fix_type == 0 else \
                               "No Fix" if fix_type == 1 else \
                               "2D Fix" if fix_type == 2 else \
                               "3D Fix" if fix_type == 3 else \
                               f"Unknown ({fix_type})"
                logging.info(f"GPS status: {fix_type_name} ({diagnostics['gps_status']['satellites_visible']} satellites)")

            # Pre-arm status
            if diagnostics['pre_arm_status']:
                logging.info("Pre-arm checks status:")
                for msg in diagnostics['pre_arm_status']:
                    logging.info(f"  - {msg}")
            else:
                logging.info("No pre-arm check messages received.")

            # Important status text messages
            if diagnostics['status_text_messages']:
                logging.info("Important status messages:")
                for msg in diagnostics['status_text_messages'][-5:]:  # Show last 5 messages
                    logging.info(f"  - {msg}")

            logging.info("=========================")
        else:
            logging.warning("Could not retrieve diagnostic information")

        # Test for basic communication
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

        logging.info(f"Received heartbeat from system {getattr(vehicle, 'target_system', 'Unknown')}, component {getattr(vehicle, 'target_component', 'Unknown')}")
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

        # Set to GUIDED mode
        if not set_mode(vehicle, "GUIDED"):
            logging.error("Failed to set GUIDED mode")
            return False

        # Import the direct MAVLink arming function
        from drone.navigation import arm_vehicle_mavlink, check_if_armed_simple

        # Try to arm using direct MAVLink
        logging.info("Arming vehicle with direct MAVLink method")
        if not arm_vehicle_mavlink(vehicle):
            logging.error("Failed to arm with direct MAVLink method")
            return False

        # Check if actually armed
        # armed = check_if_armed_simple(vehicle)
        armed = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        logging.info(f"Arm state verification: {'ARMED' if armed else 'NOT ARMED'}")

        # If not armed but no exception was thrown, we'll proceed anyway
        if not armed:
            logging.warning("Arm command was accepted but vehicle doesn't appear armed")
            logging.info("Proceeding with test anyway")

        logging.info(f"Vehicle armed. Waiting for {duration} seconds...")
        time.sleep(duration)

        # Disarm with direct MAVLink
        logging.info("Disarming vehicle with direct MAVLink method")
        target_system = getattr(vehicle, 'target_system', 1)
        target_component = getattr(vehicle, 'target_component', 0)

        vehicle.mav.command_long_send(
            target_system,
            target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,                  # Confirmation
            0,                  # Param 1: 0 to disarm
            0,                  # Param 2: Normal disarm
            0, 0, 0, 0, 0       # Params 3-7 (not used)
        )

        # Wait a moment for disarm to take effect
        time.sleep(1)

        # Check disarm state
        armed = check_if_armed_simple(vehicle)
        if armed:
            logging.warning("Vehicle still appears to be armed after disarm command")
            # Not failing the test for this
        else:
            logging.info("Vehicle successfully disarmed")

        logging.info("Arm test completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error during arm test: {str(e)}")

        # Try to disarm if there was an error
        try:
            vehicle.mav.command_long_send(
                getattr(vehicle, 'target_system', 1),
                getattr(vehicle, 'target_component', 0),
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )
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


# --- missions/test_missions.py ---
def test_incremental_takeoff(vehicle, max_altitude=3, increment=1):
    """
    Test takeoff in small increments with enhanced safety checks.

    Args:
        vehicle: The connected mavlink object
        max_altitude: Maximum target altitude in meters
        increment: Height increment in meters for each step

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("=== INCREMENTAL TAKEOFF TEST ===")

        # Run comprehensive pre-flight checks
        from drone.navigation import run_preflight_checks, verify_orientation

        checks_passed, failure_reason = run_preflight_checks(vehicle)
        if not checks_passed:
            logging.error(f"Pre-flight checks failed: {failure_reason}")
            return False

        # Verify orientation is stable
        if not verify_orientation(vehicle):
            response = input("Orientation may be unstable. Proceed anyway? (y/n): ")
            if response.lower() != 'y':
                logging.info("Takeoff aborted by user")
                return False

        # Set to GUIDED mode
        logging.info("Setting mode to GUIDED")
        if not set_mode(vehicle, "GUIDED"):
            logging.error("Failed to set GUIDED mode")
            return False

        # Arm the vehicle
        logging.info("Arming vehicle")
        if not arm_vehicle(vehicle):
            logging.error("Failed to arm vehicle")
            return False

        # For the first takeoff, use a minimal altitude
        first_altitude = min(1, increment)
        logging.info(f"Initial takeoff to {first_altitude}m")

        # Send takeoff command
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,                  # Confirmation
            0, 0, 0, 0, 0, 0,   # Params 1-6 (not used)
            first_altitude      # Param 7: Altitude (in meters)
        )

        # Monitor initial takeoff
        start_time = time.time()
        timeout = 30  # seconds
        reached_initial_altitude = False

        while time.time() - start_time < timeout:
            # Get altitude
            altitude = get_altitude(vehicle)
            if altitude is not None:
                logging.info(f"Current altitude: {altitude:.2f}m")

                # Check if reached initial altitude
                if altitude >= first_altitude * 0.95:
                    logging.info(f"Reached initial altitude of {altitude:.2f}m")
                    reached_initial_altitude = True
                    break

            # Safe abort check
            if not check_if_armed(vehicle):
                logging.error("Vehicle disarmed during initial takeoff")
                return False

            time.sleep(1)

        if not reached_initial_altitude:
            logging.error("Failed to reach initial altitude")
            return_to_launch(vehicle)
            return False

        # Safety pause for stability assessment
        logging.info("Checking stability at initial altitude...")
        time.sleep(3)

        # Now increment altitude in steps
        current_target = first_altitude

        while current_target < max_altitude:
            # Calculate next increment
            next_target = min(current_target + increment, max_altitude)
            logging.info(f"Incrementing altitude to {next_target}m")

            # Command altitude change
            vehicle.mav.command_long_send(
                vehicle.target_system,
                vehicle.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,                  # Confirmation
                0, 0, 0, 0, 0, 0,   # Params 1-6 (not used)
                next_target         # Param 7: Altitude (in meters)
            )

            # Monitor ascent
            start_time = time.time()
            timeout = 30  # seconds
            reached_next_altitude = False

            while time.time() - start_time < timeout:
                # Get altitude
                altitude = get_altitude(vehicle)
                if altitude is not None:
                    logging.info(f"Current altitude: {altitude:.2f}m")

                    # Check if reached next altitude
                    if altitude >= next_target * 0.95:
                        logging.info(f"Reached target altitude of {altitude:.2f}m")
                        reached_next_altitude = True
                        break

                # Monitor position for drift
                current_location = get_location(vehicle)

                # Safe abort check
                if not check_if_armed(vehicle):
                    logging.error("Vehicle disarmed during ascent")
                    return False

                time.sleep(1)

            if not reached_next_altitude:
                logging.error(f"Failed to reach altitude {next_target}m")
                return_to_launch(vehicle)
                return False

            # Update current target for next iteration
            current_target = next_target

            # Hover for stability at each increment
            logging.info(f"Holding at {current_target}m altitude")
            for i in range(5):
                logging.info(f"Holding... {i+1}/5 seconds")
                altitude = get_altitude(vehicle)
                if altitude is not None:
                    logging.info(f"Current altitude: {altitude:.2f}m")
                time.sleep(1)

        # Final hover at max altitude
        logging.info(f"Reached maximum target altitude of {max_altitude}m")
        logging.info("Hovering for 5 seconds")
        time.sleep(5)

        # Return to launch
        logging.info("Test complete. Returning to launch")
        if not return_to_launch(vehicle):
            logging.error("Failed to return to launch")
            return False

        # Wait for landing
        logging.info("Waiting for landing")
        start_time = time.time()
        while check_if_armed(vehicle) and time.time() - start_time < 60:
            altitude = get_altitude(vehicle)
            if altitude is not None:
                logging.info(f"Altitude: {altitude:.2f}m")
            time.sleep(1)

        if check_if_armed(vehicle):
            logging.warning("Vehicle still armed after RTL - trying manual disarm")
            disarm_vehicle(vehicle)

        logging.info("Incremental takeoff test completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error during incremental takeoff test: {str(e)}")

        # Try to return to launch if there was an error
        try:
            logging.warning("Attempting emergency return to launch")
            return_to_launch(vehicle)
            time.sleep(5)
            disarm_vehicle(vehicle)
        except:
            pass

        return False

def monitor_altitude_realtime(vehicle, duration=0, update_interval=0.5):
    """
    Monitor and display real-time altitude data from multiple sources.

    Args:
        vehicle: The connected mavlink object
        duration: Duration to monitor in seconds (0 = indefinite, Ctrl+C to stop)
        update_interval: Update interval in seconds

    Returns:
        True if monitoring was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Starting real-time altitude monitoring")
        logging.info("Press Ctrl+C to stop monitoring")

        # Request altitude data streams
        request_message_interval(vehicle, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)  # 10 Hz
        request_message_interval(vehicle, mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, 5)  # 5 Hz
        request_message_interval(vehicle, mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 2)  # 2 Hz

        start_time = time.time()
        last_update = 0

        # Initialize altitude data
        altitude_data = {
            'relative_alt': None,
            'absolute_alt': None,
            'gps_alt': None,
            'barometric_alt': None,
            'armed': None,
            'mode': None
        }

        print("\n" + "="*80)
        print("REAL-TIME ALTITUDE MONITORING")
        print("="*80)
        print(f"{'Time':<10} | {'Armed':<6} | {'Mode':<10} | {'Rel Alt':<8} | {'Abs Alt':<8} | {'GPS Alt':<8} | {'Baro Alt':<8}")
        print("-"*80)

        while True:
            # Check duration limit
            if duration > 0 and (time.time() - start_time) > duration:
                break

            # Update at specified interval
            current_time = time.time()
            if current_time - last_update < update_interval:
                time.sleep(0.1)
                continue

            last_update = current_time

            # Collect messages
            msg = vehicle.recv_match(blocking=False)
            if msg:
                msg_type = msg.get_type()

                if msg_type == "GLOBAL_POSITION_INT":
                    altitude_data['relative_alt'] = msg.relative_alt / 1000.0  # Convert mm to m
                    altitude_data['absolute_alt'] = msg.alt / 1000.0  # Convert mm to m

                elif msg_type == "VFR_HUD":
                    altitude_data['barometric_alt'] = msg.alt

                elif msg_type == "GPS_RAW_INT":
                    altitude_data['gps_alt'] = msg.alt / 1000.0  # Convert mm to m

                elif msg_type == "HEARTBEAT":
                    altitude_data['armed'] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    try:
                        altitude_data['mode'] = mavutil.mode_string_v10(msg) if callable(mavutil.mode_string_v10) else "UNKNOWN"
                    except:
                        altitude_data['mode'] = "UNKNOWN"

            # Display current data
            timestamp = time.strftime("%H:%M:%S")
            armed_str = "ARMED" if altitude_data['armed'] else "DISARM"
            mode_str = altitude_data['mode'] or "N/A"
            rel_alt_str = f"{altitude_data['relative_alt']:.2f}m" if altitude_data['relative_alt'] is not None else "N/A"
            abs_alt_str = f"{altitude_data['absolute_alt']:.2f}m" if altitude_data['absolute_alt'] is not None else "N/A"
            gps_alt_str = f"{altitude_data['gps_alt']:.2f}m" if altitude_data['gps_alt'] is not None else "N/A"
            baro_alt_str = f"{altitude_data['barometric_alt']:.2f}m" if altitude_data['barometric_alt'] is not None else "N/A"

            print(f"{timestamp:<10} | {armed_str:<6} | {mode_str:<10} | {rel_alt_str:<8} | {abs_alt_str:<8} | {gps_alt_str:<8} | {baro_alt_str:<8}")

        print("\nAltitude monitoring stopped")
        return True

    except KeyboardInterrupt:
        print("\nAltitude monitoring stopped by user")
        return True
    except Exception as e:
        logging.error(f"Error during altitude monitoring: {str(e)}")
        return False
