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
    arm_and_takeoff, check_if_armed_simple, disarm_vehicle, set_mode, get_location, get_distance_metres, navigate_to_waypoint, return_to_launch,
)

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


def wait_for_waypoint_blocking(vehicle, target_lat, target_lon, target_altitude, timeout=45, tolerance=1.0):
    """
    Blocking wait for waypoint arrival with real-time feedback and altitude monitoring.

    Args:
        vehicle: The connected mavlink object
        target_lat: Target latitude
        target_lon: Target longitude
        target_altitude: Target altitude to maintain
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
        last_altitude_correction = 0

        print(f"\nNavigating to waypoint...")
        print("Target: {:.7f}, {:.7f} at {:.1f}m".format(target_lat, target_lon, target_altitude))
        print("-" * 70)

        while time.time() - start_time < timeout:
            # Get current position
            current_location = get_location(vehicle)

            if current_location:
                current_lat, current_lon, current_alt = current_location

                # Calculate distance to target
                distance = get_distance_metres(current_location, target_location)

                # Calculate bearing for reference
                bearing = calculate_bearing(current_lat, current_lon, target_lat, target_lon)

                # Monitor altitude loss and correct if needed
                altitude_loss = target_altitude - current_alt
                if altitude_loss > 0.5 and time.time() - last_altitude_correction > 2:
                    logging.warning(f"Altitude loss detected: {altitude_loss:.2f}m, correcting...")

                    # Send altitude correction command
                    vehicle.mav.set_position_target_global_int_send(
                        0,  # time_boot_ms
                        vehicle.target_system,
                        vehicle.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b0000111111111000,  # type_mask (only alt enabled)
                        int(current_lat * 1e7),
                        int(current_lon * 1e7),
                        target_altitude,
                        0, 0, 0, 0, 0, 0, 0, 0
                    )
                    last_altitude_correction = time.time()

                # Check if within tolerance
                if distance <= tolerance:
                    stable_count += 1
                    status = f"ARRIVED ({stable_count}/{required_stable_readings})"
                else:
                    stable_count = 0
                    status = f"MOVING (bearing: {bearing:.0f}°)"

                # Real-time display with altitude
                timestamp = time.strftime("%H:%M:%S")
                alt_status = f"ALT: {current_alt:.2f}m"
                if altitude_loss > 0.3:
                    alt_status += f" (-{altitude_loss:.2f}m)"

                print(f"\r{timestamp} | Pos: {current_lat:.7f}, {current_lon:.7f} | {alt_status} | Dist: {distance:6.2f}m | {status}", end="", flush=True)

                # Check if we've reached waypoint with stability
                if stable_count >= required_stable_readings:
                    print(f"\n✓ WAYPOINT REACHED! (Final distance: {distance:.2f}m, altitude: {current_alt:.2f}m)")
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
            #(35.3481795,	-119.1046447),
            #(35.3481817,	-119.1047332),
            # (35.3481708, -119.1048297),  # South point
            (35.3481795, -119.1046386),  # East point
        ] * 70

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
            if not wait_for_waypoint_blocking(vehicle, waypoint_lat, waypoint_lon, altitude, timeout=60, tolerance=1.5):
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

def clean_message_streams(vehicle):
    """
    Clean up all existing message streams to prevent interference.
    This fixes the degradation over time issue.
    """
    if not vehicle:
        return False

    try:
        logging.info("Cleaning up message streams...")

        # Stop all message intervals
        common_message_ids = [
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
            mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT,
        ]

        for msg_id in common_message_ids:
            vehicle.mav.command_long_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, 0, 0, 0, 0, 0, 0  # 0 = stop
            )

        # Stop all legacy data streams
        for stream_id in range(13):  # ArduPilot has streams 0-12
            vehicle.mav.request_data_stream_send(
                vehicle.target_system, vehicle.target_component,
                stream_id, 0, 0  # Rate 0, stop
            )

        # Clear message buffer
        start_time = time.time()
        while time.time() - start_time < 1.0:  # Clear for 1 second
            vehicle.recv_match(blocking=False)

        logging.info("Message streams cleaned")
        return True

    except Exception as e:
        logging.warning(f"Error cleaning message streams: {str(e)}")
        return False

def setup_optimized_position_stream(vehicle, rate_hz=5):
    """
    Set up position stream optimized for your setup.
    Uses conservative rate and single method to prevent conflicts.
    """
    if not vehicle:
        return False

    try:
        # First clean existing streams
        clean_message_streams(vehicle)

        # Wait for cleanup to take effect
        time.sleep(0.5)

        logging.info(f"Setting up optimized position stream at {rate_hz} Hz")

        # Use ONLY the message interval method (which works best on your system)
        interval_us = int(1000000 / rate_hz)

        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            interval_us, 0, 0, 0, 0, 0
        )

        # Wait for acknowledgment
        ack = vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
        if ack and ack.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            if ack.result == 0:
                logging.info("Position stream setup successful")
            else:
                logging.warning(f"Position stream setup ACK result: {ack.result}")

        # Test the stream briefly
        time.sleep(0.5)
        test_msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
        if test_msg:
            lat = test_msg.lat / 1e7
            lon = test_msg.lon / 1e7
            alt = test_msg.relative_alt / 1000.0
            logging.info(f"Position stream test: {lat:.7f}, {lon:.7f}, {alt:.2f}m")
            return True
        else:
            logging.warning("Position stream test failed")
            return False

    except Exception as e:
        logging.error(f"Error setting up position stream: {str(e)}")
        return False

def get_location_single_request(vehicle, timeout=2):
    """
    Get location with single clean request - no sustained streaming.
    This prevents the degradation issue you're experiencing.
    """
    if not vehicle:
        return None

    try:
        # Clear buffer first
        while vehicle.recv_match(blocking=False):
            pass

        # Request ONE position update
        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            100000, 0, 0, 0, 0, 0  # 10 Hz for just this request
        )

        # Get the message
        msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)

        # Stop the stream immediately
        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            0, 0, 0, 0, 0, 0  # Stop
        )

        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0
            return (lat, lon, alt)
        else:
            return None

    except Exception as e:
        logging.warning(f"Error getting single location: {str(e)}")
        return None

def wait_for_waypoint_optimized(vehicle, target_lat, target_lon, target_altitude, timeout=60, tolerance=2.0):
    """
    Optimized waypoint waiting that works with your message rate patterns.
    Uses burst requests instead of sustained streaming.
    """
    if not vehicle:
        return False

    try:
        logging.info(f"Optimized navigation to: {target_lat:.7f}, {target_lon:.7f}")

        start_time = time.time()
        target_location = (target_lat, target_lon, 0)
        consecutive_good = 0
        required_good = 3
        last_distance = None
        check_interval = 1.0  # Check position every 1 second
        last_check = 0

        print(f"\nOptimized waypoint navigation...")
        print("Target: {:.7f}, {:.7f} at {:.1f}m".format(target_lat, target_lon, target_altitude))
        print("-" * 70)

        while time.time() - start_time < timeout:
            current_time = time.time()

            # Only check position at intervals to prevent stream degradation
            if current_time - last_check >= check_interval:
                # Get position with single clean request
                current_location = get_location_single_request(vehicle, timeout=2)
                last_check = current_time

                if current_location:
                    current_lat, current_lon, current_alt = current_location

                    # Calculate distance
                    distance = get_distance_metres(current_location, target_location)
                    bearing = calculate_bearing(current_lat, current_lon, target_lat, target_lon)

                    # Check altitude and correct if needed
                    altitude_error = target_altitude - current_alt
                    if abs(altitude_error) > 0.8:
                        logging.info(f"Altitude correction needed: {altitude_error:+.2f}m")

                        try:
                            vehicle.mav.set_position_target_global_int_send(
                                0, vehicle.target_system, vehicle.target_component,
                                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                0b0000111111111000,  # Only altitude
                                int(current_lat * 1e7), int(current_lon * 1e7), target_altitude,
                                0, 0, 0, 0, 0, 0, 0, 0
                            )
                        except Exception as e:
                            logging.warning(f"Altitude correction failed: {str(e)}")

                    # Check if within tolerance
                    if distance <= tolerance:
                        consecutive_good += 1
                        status = f"ARRIVED ({consecutive_good}/{required_good})"
                    else:
                        consecutive_good = 0

                        # Show progress
                        if last_distance and distance < last_distance:
                            status = f"APPROACHING (↓{last_distance-distance:.1f}m, bearing {bearing:.0f}°)"
                        else:
                            status = f"MOVING (bearing {bearing:.0f}°)"

                    # Display with timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    alt_display = f"ALT: {current_alt:.2f}m"
                    if abs(altitude_error) > 0.3:
                        alt_display += f" ({altitude_error:+.2f}m)"

                    print(f"{timestamp} | Pos: {current_lat:.7f}, {current_lon:.7f} | {alt_display} | Dist: {distance:6.2f}m | {status}")

                    # Check if waypoint reached
                    if consecutive_good >= required_good:
                        print(f"✓ WAYPOINT REACHED! (Final: {distance:.2f}m, alt: {current_alt:.2f}m)")
                        return True

                    last_distance = distance

                else:
                    print(f"{time.strftime('%H:%M:%S')} | ⚠️  Position request failed")

            # Brief sleep between checks
            time.sleep(0.2)

        print(f"❌ Timeout after {timeout}s (final distance: {last_distance:.2f}m)" if last_distance else f"❌ Timeout after {timeout}s")
        return False

    except Exception as e:
        logging.error(f"Error during optimized waypoint wait: {str(e)}")
        return False

def command_waypoint_clean(vehicle, target_lat, target_lon, altitude):
    """Send waypoint command with clean approach"""
    if not vehicle:
        return False

    try:
        logging.info(f"Commanding waypoint: {target_lat:.7f}, {target_lon:.7f} at {altitude}m")

        vehicle.mav.set_position_target_global_int_send(
            0, vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # Position only
            int(target_lat * 1e7), int(target_lon * 1e7), altitude,
            0, 0, 0, 0, 0, 0, 0, 0
        )

        time.sleep(0.1)
        return True

    except Exception as e:
        logging.error(f"Error sending waypoint: {str(e)}")
        return False

def mission_diamond_precision_fixed(vehicle, altitude=5, loops=1, mission_file=None):
    """
    Diamond mission optimized for your specific setup with mission file support.
    Fixes the message degradation issue.

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters (overrides mission file altitudes if provided)
        loops: Number of loops to repeat the waypoints
        mission_file: Path to Mission Planner .mission file (optional)
    """
    if not vehicle:
        return False

    try:
        logging.info("=== OPTIMIZED DIAMOND MISSION (Fixed for Your Setup) ===")

        # Clean up any existing streams first
        clean_message_streams(vehicle)

        # Determine waypoints source
        if mission_file:
            logging.info(f"Loading waypoints from mission file: {mission_file}")

            # Import mission parser
            from missions.mission_parser import MissionParser

            # Parse mission file
            parser = MissionParser()
            if not parser.parse_mission_file(mission_file):
                logging.error("Failed to parse mission file")
                return False

            # Get simple waypoints (lat, lon pairs)
            mission_waypoints = parser.get_simple_waypoints()

            if not mission_waypoints:
                logging.error("No navigation waypoints found in mission file")
                return False

            # Use mission waypoints, repeated by loops
            diamond_waypoints = mission_waypoints * loops

            logging.info(f"Loaded {len(mission_waypoints)} waypoints from mission file")
            logging.info(f"Total waypoints with {loops} loops: {len(diamond_waypoints)}")

            # If altitude parameter provided, it overrides mission file altitudes
            # (Mission file altitudes are ignored when using simple waypoints)
            logging.info(f"Using altitude: {altitude}m (overrides mission file altitudes)")

        else:
            logging.info("Using default hardcoded waypoints")
            # Your original waypoints
            diamond_waypoints = [
                (35.3481866, -119.1047372),  # left
                (35.3481888, -119.1048713),  # right
            ] * loops

        logging.info(f"Mission will visit {len(diamond_waypoints)} waypoints at {altitude}m altitude")

        # Pre-flight checks and takeoff (using your existing functions)
        from drone.navigation import run_preflight_checks, set_mode, arm_and_takeoff, return_to_launch, check_if_armed, disarm_vehicle

        checks_passed, failure_reason = run_preflight_checks(vehicle)
        if not checks_passed:
            logging.error(f"Pre-flight checks failed: {failure_reason}")
            return False

        if not set_mode(vehicle, "GUIDED"):
            logging.error("Failed to set GUIDED mode")
            return False

        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to takeoff")
            return False

        # Get home position
        home_location = get_location_single_request(vehicle, timeout=5)
        if home_location:
            logging.info(f"Home: {home_location[0]:.7f}, {home_location[1]:.7f}")

        # Navigate to waypoints
        for i, (lat, lon) in enumerate(diamond_waypoints, 1):
            logging.info(f"\n📍 WAYPOINT {i}/{len(diamond_waypoints)}")

            if not command_waypoint_clean(vehicle, lat, lon, altitude):
                logging.error(f"Failed to command waypoint {i}")
                return_to_launch(vehicle)
                return False

            if not wait_for_waypoint_optimized(vehicle, lat, lon, altitude, timeout=90, tolerance=2.0):
                logging.error(f"Failed to reach waypoint {i}")
                return_to_launch(vehicle)
                return False

            logging.info(f"✓ Waypoint {i} complete")
            if i < len(diamond_waypoints):
                time.sleep(2)

        # Return home
        logging.info("\n🏠 RETURNING HOME")
        return_to_launch(vehicle)

        # Wait for landing
        start_time = time.time()
        while time.time() - start_time < 120:
            if not check_if_armed(vehicle):
                logging.info("✓ Landed and disarmed")
                break
            time.sleep(2)

        # Final cleanup
        clean_message_streams(vehicle)

        logging.info("🎉 OPTIMIZED MISSION COMPLETE")
        return True

    except Exception as e:
        logging.error(f"Mission error: {str(e)}")
        try:
            clean_message_streams(vehicle)
            return_to_launch(vehicle)
        except:
            pass
        return False
