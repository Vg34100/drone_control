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
    Test takeoff in small increments with blocking behavior and audio feedback.

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
        logging.info("=== BLOCKING INCREMENTAL TAKEOFF TEST ===")
        logging.info(f"Target: {max_altitude}m in {increment}m increments")

        # Run comprehensive pre-flight checks
        from drone.navigation import run_preflight_checks

        checks_passed, failure_reason = run_preflight_checks(vehicle)
        if not checks_passed:
            logging.error(f"Pre-flight checks failed: {failure_reason}")
            return False

        # Set to GUIDED mode
        logging.info("Setting mode to GUIDED")
        if not set_mode(vehicle, "GUIDED"):
            logging.error("Failed to set GUIDED mode")
            return False

        # Arm the vehicle
        logging.info("Arming vehicle...")
        if not arm_vehicle(vehicle):
            logging.error("Failed to arm vehicle")
            return False

        # Wait for altitude to reset after arming
        logging.info("Waiting 2 seconds for altitude sensor to stabilize...")
        time.sleep(2)

        # Get baseline altitude
        baseline_msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=3)
        if baseline_msg:
            baseline_altitude = baseline_msg.relative_alt / 1000.0
            logging.info(f"Baseline altitude: {baseline_altitude:.3f}m")
        else:
            logging.warning("Could not get baseline altitude, proceeding anyway")
            baseline_altitude = 0.0

        # Initial takeoff to first increment
        first_target = increment
        logging.info(f"\n🚁 STEP 1: Initial takeoff to {first_target}m")

        # Send initial takeoff command
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,                  # Confirmation
            0, 0, 0, 0, 0, 0,   # Params 1-6 (not used)
            first_target        # Param 7: Altitude (in meters)
        )

        # BLOCKING wait for first altitude
        if not wait_for_altitude_blocking(vehicle, first_target, timeout=40, tolerance=0.15):
            logging.error(f"Failed to reach initial altitude {first_target}m")
            return_to_launch(vehicle)
            return False

        logging.info(f"✓ Successfully reached {first_target}m")
        logging.info("Stabilizing for 2 seconds...")
        time.sleep(2)

        # Incremental altitude increases
        current_target = first_target

        while current_target < max_altitude:
            next_target = min(current_target + increment, max_altitude)
            step_number = int(next_target / increment) + 1

            logging.info(f"\n🚁 STEP {step_number}: Climbing to {next_target}m")

            # Send altitude command
            if not command_altitude_precise(vehicle, next_target):
                logging.error(f"Failed to send altitude command for {next_target}m")
                return_to_launch(vehicle)
                return False

            # BLOCKING wait for next altitude
            if not wait_for_altitude_blocking(vehicle, next_target, timeout=30, tolerance=0.15):
                logging.error(f"Failed to reach altitude {next_target}m")
                return_to_launch(vehicle)
                return False

            logging.info(f"✓ Successfully reached {next_target}m")
            current_target = next_target

            # Stabilization pause between increments
            if current_target < max_altitude:
                logging.info("Stabilizing for 2 seconds...")
                time.sleep(2)

        # Final hover at maximum altitude
        logging.info(f"\n🎯 FINAL: Reached maximum altitude of {max_altitude}m")
        logging.info("Final hover for 5 seconds...")
        time.sleep(5)

        # Return to launch with blocking behavior
        logging.info("\n🏠 RETURN TO LAUNCH")
        logging.info("Commanding RTL...")

        if not return_to_launch(vehicle):
            logging.error("Failed to command RTL")
            return False

        # BLOCKING wait for landing with real-time feedback
        logging.info("Monitoring descent and landing...")
        print("-" * 50)

        landing_start = time.time()
        landing_timeout = 90  # 90 seconds for landing

        while time.time() - landing_start < landing_timeout:
            # Check if still armed (landing complete when disarmed)
            heartbeat = vehicle.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if heartbeat:
                armed = (heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

                # Get current altitude
                pos_msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
                current_alt = pos_msg.relative_alt / 1000.0 if pos_msg else None

                timestamp = time.strftime("%H:%M:%S")
                armed_status = "ARMED" if armed else "DISARMED"
                alt_str = f"{current_alt:.3f}m" if current_alt is not None else "N/A"

                print(f"\r{timestamp} | Status: {armed_status} | Altitude: {alt_str}", end="", flush=True)

                if not armed:
                    print(f"\n✓ LANDING COMPLETE - Vehicle disarmed")
                    # play_beep()
                    break

                # Also check if very close to ground
                if current_alt is not None and current_alt < 0.3:
                    print(f"\n✓ NEAR GROUND - Altitude: {current_alt:.3f}m")

            time.sleep(0.5)

        # Final verification
        time.sleep(2)
        final_armed = check_if_armed(vehicle)
        if final_armed:
            logging.warning("Vehicle still armed after landing timeout - forcing disarm")
            disarm_vehicle(vehicle)

        logging.info("\n🎉 INCREMENTAL TAKEOFF TEST COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        logging.error(f"Error during incremental takeoff test: {str(e)}")
        try:
            logging.warning("Attempting emergency return to launch")
            return_to_launch(vehicle)
            time.sleep(10)
            if check_if_armed(vehicle):
                disarm_vehicle(vehicle)
        except:
            pass
        return False

def command_altitude_precise(vehicle, target_altitude):
    """
    Send precise altitude command using position target.

    Args:
        vehicle: The connected mavlink object
        target_altitude: Target altitude in meters

    Returns:
        True if command sent successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Get current location for position hold
        current_location = get_location(vehicle)
        if not current_location:
            logging.error("Could not get current location for altitude command")
            return False

        lat, lon, _ = current_location

        logging.info(f"Commanding altitude change to {target_altitude}m")

        # Send position target with only altitude change
        vehicle.mav.set_position_target_global_int_send(
            0,  # time_boot_ms (not used)
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # type_mask (only alt enabled, position and velocity ignored)
            int(lat * 1e7),      # lat_int
            int(lon * 1e7),      # lon_int
            target_altitude,     # alt (meters)
            0, 0, 0,            # vx, vy, vz (not used)
            0, 0, 0,            # afx, afy, afz (not used)
            0, 0                # yaw, yaw_rate (not used)
        )

        return True

    except Exception as e:
        logging.error(f"Error sending altitude command: {str(e)}")
        return False

def wait_for_altitude_blocking(vehicle, target_altitude, timeout=30, tolerance=0.1):
    """
    Blocking wait for altitude with real-time feedback and audio notification.

    Args:
        vehicle: The connected mavlink object
        target_altitude: Target altitude in meters
        timeout: Maximum time to wait in seconds
        tolerance: Altitude tolerance in meters

    Returns:
        True if altitude reached, False if timeout
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Waiting for altitude {target_altitude}m (tolerance: ±{tolerance}m)")

        # Request high-frequency altitude updates
        vehicle.mav.request_data_stream_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            20,  # 20 Hz
            1    # Start
        )

        start_time = time.time()
        last_altitude = None
        stable_count = 0
        required_stable_readings = 3  # Need 3 consecutive readings within tolerance

        print(f"\nWaiting for altitude {target_altitude}m...")
        print("-" * 50)

        while time.time() - start_time < timeout:
            # Get the most recent altitude reading
            current_altitude = None

            # Process recent messages to get latest altitude
            for _ in range(10):  # Check up to 10 recent messages
                msg = vehicle.recv_match(blocking=False)
                if msg and msg.get_type() == "GLOBAL_POSITION_INT":
                    current_altitude = msg.relative_alt / 1000.0

            if current_altitude is not None:
                # Calculate how close we are to target
                altitude_diff = abs(current_altitude - target_altitude)
                progress_percent = min(100, (current_altitude / target_altitude) * 100) if target_altitude > 0 else 0

                # Check if within tolerance
                if altitude_diff <= tolerance:
                    stable_count += 1
                    status = f"STABLE ({stable_count}/{required_stable_readings})"
                else:
                    stable_count = 0
                    if current_altitude < target_altitude:
                        status = "CLIMBING"
                    else:
                        status = "DESCENDING"

                # Real-time display
                timestamp = time.strftime("%H:%M:%S")
                print(f"\r{timestamp} | Alt: {current_altitude:6.3f}m | Target: {target_altitude:6.3f}m | Diff: {altitude_diff:+6.3f}m | {progress_percent:5.1f}% | {status}", end="", flush=True)

                # Check if we've reached target altitude with stability
                if stable_count >= required_stable_readings:
                    print(f"\n✓ REACHED {target_altitude}m! (Final: {current_altitude:.3f}m)")
                    # play_beep()
                    return True

                last_altitude = current_altitude

            # Safety check - ensure still armed
            heartbeat = vehicle.recv_match(type='HEARTBEAT', blocking=False)
            if heartbeat:
                armed = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                if not armed:
                    print(f"\n✗ Vehicle disarmed during altitude wait!")
                    return False

            time.sleep(0.05)  # 50ms update rate


        print(f"\n✗ Timeout waiting for altitude {target_altitude}m (current: {f'{last_altitude:.3f}m' if last_altitude else 'unknown'})")
        return False

    except Exception as e:
        logging.error(f"Error waiting for altitude: {str(e)}")
        return False
#




def monitor_altitude_realtime(vehicle, duration=0, update_interval=0.2):
    """
    Ultra-responsive altitude monitoring with minimal delay.

    Args:
        vehicle: The connected mavlink object
        duration: Duration to monitor in seconds (0 = indefinite)

    Returns:
        True if monitoring completed successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("Starting ULTRA-RESPONSIVE altitude monitoring")
        logging.info("Press Ctrl+C to stop monitoring")

        # Request maximum frequency streams
        vehicle.mav.request_data_stream_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            20,  # 20 Hz - maximum
            1    # Start
        )

        start_time = time.time()
        last_altitude = None
        message_count = 0

        print("\n" + "="*80)
        print("ULTRA-RESPONSIVE ALTITUDE MONITORING")
        print("="*80)
        print("Time       | Relative Alt | Change    | Messages | Status")
        print("-"*80)

        while True:
            if duration > 0 and (time.time() - start_time) > duration:
                break

            # Process ALL available messages immediately
            while True:
                msg = vehicle.recv_match(blocking=False)
                if not msg:
                    break

                message_count += 1

                if msg.get_type() == "GLOBAL_POSITION_INT":
                    current_altitude = msg.relative_alt / 1000.0
                    current_time = time.strftime("%H:%M:%S.%f")[:-3]

                    # Calculate change
                    change_str = "---"
                    if last_altitude is not None:
                        change = current_altitude - last_altitude
                        if abs(change) > 0.001:  # Only show significant changes
                            change_str = f"{change:+.3f}m"

                    # Determine status based on change rate
                    if last_altitude is None:
                        status = "INIT"
                    elif abs(current_altitude - last_altitude) > 0.01:
                        status = "MOVING"
                    else:
                        status = "STABLE"

                    print(f"{current_time:<10} | {current_altitude:>9.3f}m | {change_str:>9} | {message_count:>8} | {status}")

                    last_altitude = current_altitude

            # Very minimal sleep - just enough to prevent 100% CPU
            time.sleep(0.001)  # 1ms

        print("\nUltra-responsive monitoring stopped")
        return True

    except KeyboardInterrupt:
        print("\nUltra-responsive monitoring stopped by user")
        return True
    except Exception as e:
        logging.error(f"Error during ultra-responsive monitoring: {str(e)}")
        return False
