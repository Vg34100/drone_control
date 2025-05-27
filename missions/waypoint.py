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
    arm_and_takeoff, check_if_armed_simple, disarm_vehicle, set_mode, get_location, get_distance_metres,
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


def wait_for_waypoint_blocking(vehicle, target_lat, target_lon, timeout=45, tolerance=1.0):
    """
    Blocking wait for waypoint arrival with real-time feedback.

    Args:
        vehicle: The connected mavlink object
        target_lat: Target latitude
        target_lon: Target longitude
        timeout: Maximum time to wait in seconds
        tolerance: Distance tolerance in meters

    Returns:
        True if waypoint reached, False if timeout
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Navigating to waypoint: {target_lat:.7f}, {target_lon:.7f}")

        # Request high-frequency position updates
        vehicle.mav.request_data_stream_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            10,  # 10 Hz
            1    # Start
        )

        start_time = time.time()
        last_distance = None
        stable_count = 0
        required_stable_readings = 3
        target_location = (target_lat, target_lon, 0)

        print(f"\nNavigating to waypoint...")
        print("Target: {:.7f}, {:.7f}".format(target_lat, target_lon))
        print("-" * 60)

        while time.time() - start_time < timeout:
            # Get current position
            current_location = get_location(vehicle)

            if current_location:
                current_lat, current_lon, current_alt = current_location

                # Calculate distance to target
                distance = get_distance_metres(current_location, target_location)

                # Calculate bearing for reference
                bearing = calculate_bearing(current_lat, current_lon, target_lat, target_lon)

                # Check if within tolerance
                if distance <= tolerance:
                    stable_count += 1
                    status = f"ARRIVED ({stable_count}/{required_stable_readings})"
                else:
                    stable_count = 0
                    status = f"MOVING (bearing: {bearing:.0f}°)"

                # Real-time display
                timestamp = time.strftime("%H:%M:%S")
                print(f"\r{timestamp} | Pos: {current_lat:.7f}, {current_lon:.7f} | Dist: {distance:6.2f}m | {status}", end="", flush=True)

                # Check if we've reached waypoint with stability
                if stable_count >= required_stable_readings:
                    print(f"\n✓ WAYPOINT REACHED! (Final distance: {distance:.2f}m)")
                    return True

                last_distance = distance

            # Safety check - ensure still armed and in correct mode
            heartbeat = vehicle.recv_match(type='HEARTBEAT', blocking=False)
            if heartbeat:
                armed = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                if not armed:
                    print(f"\n✗ Vehicle disarmed during waypoint navigation!")
                    return False

            time.sleep(0.2)  # 200ms update rate

        print(f"\n✗ Timeout reaching waypoint (final distance: {last_distance:.2f}m)" if last_distance else "\n✗ Timeout reaching waypoint")
        return False

    except Exception as e:
        logging.error(f"Error waiting for waypoint: {str(e)}")
        return False

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2.

    Returns:
        Bearing in degrees (0-360)
    """
    import math

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)

    return (bearing_deg + 360) % 360

def command_waypoint_precise(vehicle, target_lat, target_lon, altitude):
    """
    Send precise waypoint command using position target.

    Args:
        vehicle: The connected mavlink object
        target_lat: Target latitude
        target_lon: Target longitude
        altitude: Target altitude in meters

    Returns:
        True if command sent successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Commanding waypoint: {target_lat:.7f}, {target_lon:.7f} at {altitude}m")

        # Send position target
        vehicle.mav.set_position_target_global_int_send(
            0,  # time_boot_ms (not used)
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # type_mask (position only)
            int(target_lat * 1e7),  # lat_int
            int(target_lon * 1e7),  # lon_int
            altitude,               # alt (meters)
            0, 0, 0,               # vx, vy, vz (not used)
            0, 0, 0,               # afx, afy, afz (not used)
            0, 0                   # yaw, yaw_rate (not used)
        )

        return True

    except Exception as e:
        logging.error(f"Error sending waypoint command: {str(e)}")
        return False

def mission_diamond_precision(vehicle, altitude=5):
    """
    Execute a precision diamond waypoint mission with blocking behavior.

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters

    Returns:
        True if mission was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("=== PRECISION DIAMOND WAYPOINT MISSION ===")
        logging.info(f"Flight altitude: {altitude}m")

        # Define diamond waypoints around your field
        diamond_waypoints = [
            # (35.3482145, -119.1048425),  # North point
            # (35.3482019, -119.1049813),  # West point
            (35.3481850,	-119.1049075), # New West
            # (35.3481708, -119.1048297),  # South point
            (35.3481795, -119.1046386),  # East point
        ]

        # Run pre-flight checks
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

        # Arm and takeoff
        logging.info(f"🚁 TAKEOFF: Arming and taking off to {altitude}m")
        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to arm and takeoff")
            return False

        # Get home location for reference
        home_location = get_location(vehicle)
        if home_location:
            home_lat, home_lon, _ = home_location
            logging.info(f"Home position: {home_lat:.7f}, {home_lon:.7f}")
        else:
            logging.warning("Could not get home location")

        # Navigate to each waypoint in the diamond
        for i, (waypoint_lat, waypoint_lon) in enumerate(diamond_waypoints, 1):
            logging.info(f"\n📍 WAYPOINT {i}/{len(diamond_waypoints)}: Diamond Point {i}")

            # Send waypoint command
            if not command_waypoint_precise(vehicle, waypoint_lat, waypoint_lon, altitude):
                logging.error(f"Failed to send waypoint {i} command")
                return_to_launch(vehicle)
                return False

            # BLOCKING wait for waypoint arrival
            if not wait_for_waypoint_blocking(vehicle, waypoint_lat, waypoint_lon, timeout=60, tolerance=1.5):
                logging.error(f"Failed to reach waypoint {i}")
                return_to_launch(vehicle)
                return False

            logging.info(f"✓ Successfully reached waypoint {i}")

            # Brief pause at each waypoint (except the last one)
            if i < len(diamond_waypoints):
                logging.info("Stabilizing for 2 seconds...")
                time.sleep(2)

        # Quick pause at final waypoint
        logging.info(f"\n🎯 DIAMOND COMPLETE: All {len(diamond_waypoints)} waypoints reached")
        logging.info("Final stabilization for 1 second...")
        time.sleep(1)

        # Return to launch with blocking behavior
        logging.info("\n🏠 RETURN TO LAUNCH")
        logging.info("Commanding RTL...")

        if not return_to_launch(vehicle):
            logging.error("Failed to command RTL")
            return False

        # BLOCKING wait for landing with real-time feedback
        logging.info("Monitoring return and landing...")
        print("-" * 50)

        landing_start = time.time()
        landing_timeout = 90

        while time.time() - landing_start < landing_timeout:
            # Check armed status and altitude
            heartbeat = vehicle.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if heartbeat:
                armed = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

                # Get current position and altitude
                pos_msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
                if pos_msg:
                    current_alt = pos_msg.relative_alt / 1000.0
                    current_lat = pos_msg.lat / 1e7
                    current_lon = pos_msg.lon / 1e7

                    # Calculate distance to home if we have home location
                    dist_to_home = "N/A"
                    if home_location:
                        current_pos = (current_lat, current_lon, current_alt)
                        dist_to_home = f"{get_distance_metres(current_pos, home_location):.1f}m"
                else:
                    current_alt = None
                    dist_to_home = "N/A"

                timestamp = time.strftime("%H:%M:%S")
                armed_status = "ARMED" if armed else "DISARMED"
                alt_str = f"{current_alt:.3f}m" if current_alt is not None else "N/A"

                print(f"\r{timestamp} | Status: {armed_status} | Alt: {alt_str} | Home Dist: {dist_to_home}", end="", flush=True)

                if not armed:
                    print(f"\n✓ LANDING COMPLETE - Vehicle disarmed")
                    break

            time.sleep(0.5)

        # Final verification
        time.sleep(1)
        final_armed = check_if_armed_simple(vehicle)
        if final_armed:
            logging.warning("Vehicle still armed after landing timeout - forcing disarm")
            disarm_vehicle(vehicle)

        logging.info("\n🎉 DIAMOND WAYPOINT MISSION COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        logging.error(f"Error during diamond waypoint mission: {str(e)}")
        try:
            logging.warning("Attempting emergency return to launch")
            return_to_launch(vehicle)
            time.sleep(10)
            if check_if_armed_simple(vehicle):
                disarm_vehicle(vehicle)
        except:
            pass
        return False
