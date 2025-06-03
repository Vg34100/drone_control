# missions/waypoint_bullseye.py - COMPLETE REWRITE v2
"""
Bullseye Detection with Collection and Middle Point Landing
---------------------------------------------------------
Collects all bullseye detections during flight, then lands at the middle point.
"""

import logging
import time
import threading
import cv2
import math
from pymavlink import mavutil

from drone.connection import get_vehicle_state
from drone.navigation import (
    run_preflight_checks, set_mode, arm_and_takeoff, return_to_launch,
    check_if_armed, disarm_vehicle, get_location, send_ned_velocity,
    get_distance_metres, get_location_metres
)
from detection.bullseye_detector import BullseyeDetector
from detection.camera import initialize_camera, capture_frame, close_camera

def mission_waypoint_bullseye_detection(vehicle, altitude=6, model_path="models/best.pt",
                                      confidence=0.5, loops=1, land_on_detection=True,
                                      video_recorder=None):
    """
    Execute waypoint mission collecting all bullseye detections and landing at middle point.

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters (default 6m)
        model_path: Path to YOLO model
        confidence: Detection confidence threshold
        loops: Number of times to repeat waypoint pattern
        land_on_detection: Whether to land when bullseye is detected
        video_recorder: Existing video recorder to share camera (optional)

    Returns:
        True if mission completed successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("=== BULLSEYE COLLECTION AND MIDDLE POINT LANDING MISSION ===")
        logging.info(f"Flight altitude: {altitude}m")
        logging.info(f"Loops: {loops}")

        # Your specific waypoints
        waypoints = [
            (35.3481828, -119.1049256),  # Point 1
            (35.3481833, -119.1046789),  # Point 2
        ] * loops

        # Initialize bullseye detector
        detector = BullseyeDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            imgsz=160
        )

        # Collection of ALL detection locations
        all_detections = []
        shared_camera = video_recorder is not None

        # Run pre-flight checks
        checks_passed, failure_reason = run_preflight_checks(vehicle)
        if not checks_passed:
            logging.error(f"Pre-flight checks failed: {failure_reason}")
            return False

        # Set to GUIDED mode and takeoff
        if not set_mode(vehicle, "GUIDED"):
            logging.error("Failed to set GUIDED mode")
            return False

        if not arm_and_takeoff(vehicle, altitude):
            logging.error("Failed to arm and takeoff")
            return False

        # Fly waypoints while collecting ALL detections
        for i, (target_lat, target_lon) in enumerate(waypoints, 1):
            logging.info(f"\n📍 Flying to waypoint {i}: {target_lat:.7f}, {target_lon:.7f}")

            # Send waypoint command
            command_waypoint(vehicle, target_lat, target_lon, altitude)

            # Monitor flight path and collect ALL detections
            path_detections = collect_detections_during_flight(
                vehicle, target_lat, target_lon, altitude, detector, video_recorder
            )

            # Add to our collection
            all_detections.extend(path_detections)

            if path_detections:
                logging.info(f"🎯 Collected {len(path_detections)} detections on path to waypoint {i}")
            else:
                logging.info(f"📍 No detections on path to waypoint {i}")

        # Process collected detections
        if len(all_detections) > 0:
            logging.info(f"🎯 TOTAL DETECTIONS COLLECTED: {len(all_detections)}")

            # Calculate middle point of all detections
            middle_point = calculate_middle_detection_point(all_detections)

            if middle_point:
                logging.info(f"📍 Calculated middle detection point: {middle_point[0]:.7f}, {middle_point[1]:.7f}")

                # Execute landing at middle point
                success = execute_middle_point_landing(
                    vehicle, middle_point, detector, video_recorder, altitude
                )

                if success:
                    logging.info("🎉 MISSION SUCCESS: Landed at middle of all detections!")
                    return True
                else:
                    logging.error("❌ Middle point landing failed")
            else:
                logging.error("❌ Could not calculate middle point")
        else:
            logging.info("📍 No bullseyes detected during entire mission")

        # Return to launch if no detections or landing failed
        logging.info("\n🏠 RETURN TO LAUNCH")
        return_to_launch(vehicle)
        wait_for_landing(vehicle)

        logging.info("🎉 WAYPOINT COLLECTION MISSION COMPLETED")
        return True

    except Exception as e:
        logging.error(f"Error during waypoint collection mission: {str(e)}")
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False

def collect_detections_during_flight(vehicle, target_lat, target_lon, altitude, detector, video_recorder):
    """
    Collect ALL bullseye detections during flight to waypoint.
    Returns list of detection dictionaries with GPS coordinates and camera info.
    """
    try:
        detections = []
        target_location = (target_lat, target_lon, 0)
        detection_check_interval = 0.1  # Check every 100ms for maximum coverage
        last_check = 0

        logging.info("🔍 Collecting detections during flight...")

        while True:
            current_time = time.time()

            # Frequent detection checks
            if current_time - last_check >= detection_check_interval:
                # Get EXACT current GPS location
                current_location = get_location(vehicle)
                if not current_location:
                    time.sleep(0.05)
                    continue

                # Get camera frame at EXACT same time
                frame = get_camera_frame(video_recorder)
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Check for bullseye in frame
                bullseyes, _ = detector.detect_bullseyes_in_frame(frame)

                if len(bullseyes) > 0:
                    # Bullseye detected! Store EXACT GPS-to-camera link
                    best_detection = max(bullseyes, key=lambda x: x[3])
                    center_x, center_y, bbox_info, confidence = best_detection

                    detection_data = {
                        'gps_location': current_location,  # EXACT GPS when frame captured
                        'camera_center': (center_x, center_y),
                        'confidence': confidence,
                        'timestamp': current_time
                    }

                    detections.append(detection_data)

                    logging.info(f"🎯 Detection #{len(detections)}: GPS {current_location[0]:.7f}, {current_location[1]:.7f} | "
                               f"Camera ({center_x}, {center_y}) | Conf: {confidence:.3f}")

                # Check if we've reached the waypoint
                distance = get_distance_metres(current_location, target_location)
                if distance <= 1.5:  # Within 1.5 meters
                    logging.info(f"✅ Reached waypoint (distance: {distance:.1f}m)")
                    break

                last_check = current_time

            time.sleep(0.02)  # Very small sleep for maximum detection coverage

        logging.info(f"📊 Collected {len(detections)} detections on this path segment")
        return detections

    except Exception as e:
        logging.error(f"Error during detection collection: {str(e)}")
        return []

def calculate_middle_detection_point(all_detections):
    """
    Calculate the middle GPS point of all detections.
    Returns (lat, lon, alt) of the middle point.
    """
    try:
        if len(all_detections) == 0:
            return None

        # Extract all GPS coordinates
        latitudes = [d['gps_location'][0] for d in all_detections]
        longitudes = [d['gps_location'][1] for d in all_detections]
        altitudes = [d['gps_location'][2] for d in all_detections]

        # Calculate middle point (arithmetic mean)
        middle_lat = sum(latitudes) / len(latitudes)
        middle_lon = sum(longitudes) / len(longitudes)
        middle_alt = sum(altitudes) / len(altitudes)

        logging.info(f"📊 Detection spread:")
        logging.info(f"   Lat range: {min(latitudes):.7f} to {max(latitudes):.7f}")
        logging.info(f"   Lon range: {min(longitudes):.7f} to {max(longitudes):.7f}")
        logging.info(f"   Middle point: {middle_lat:.7f}, {middle_lon:.7f}")

        return (middle_lat, middle_lon, middle_alt)

    except Exception as e:
        logging.error(f"Error calculating middle point: {str(e)}")
        return None

def execute_middle_point_landing(vehicle, middle_point, detector, video_recorder, original_altitude):
    """
    Execute landing sequence at the calculated middle point.
    """
    try:
        logging.info("🎯 EXECUTING MIDDLE POINT LANDING SEQUENCE")

        middle_lat, middle_lon, middle_alt = middle_point

        # Step 1: Navigate to middle point
        logging.info(f"📍 Step 1: Flying to middle detection point {middle_lat:.7f}, {middle_lon:.7f}")

        command_waypoint(vehicle, middle_lat, middle_lon, original_altitude)

        # BLOCKING wait for arrival
        if not wait_for_waypoint_blocking(vehicle, middle_lat, middle_lon, original_altitude, timeout=30, tolerance=1.5):
            logging.error("❌ Failed to reach middle point")
            return False

        logging.info("✅ Reached middle detection point")

        # Step 2: Verify bullseye at middle point
        frame = get_camera_frame(video_recorder)
        if frame is not None:
            bullseyes, _ = detector.detect_bullseyes_in_frame(frame)
            if len(bullseyes) == 0:
                logging.warning("⚠️ No bullseye at middle point - searching in small circle")

                # Small circle search at middle point
                found_location = search_in_circle(vehicle, middle_point, detector, video_recorder, radius=0.5)
                if found_location:
                    middle_point = found_location
                    middle_lat, middle_lon, middle_alt = middle_point
                    logging.info("🎯 Found bullseye in circle search")
                else:
                    logging.warning("⚠️ No bullseye in circle - proceeding with original middle point")

        # Step 3: Climb for better detection view
        detection_altitude = original_altitude + 2
        logging.info(f"📈 Step 3: Climbing to {detection_altitude}m for precision landing")

        command_waypoint(vehicle, middle_lat, middle_lon, detection_altitude)

        if not wait_for_altitude_blocking(vehicle, detection_altitude, timeout=20, tolerance=0.3):
            logging.warning("⚠️ Climb timeout - continuing anyway")

        # Step 4: Center on bullseye with drone geometry compensation
        logging.info("🎯 Step 4: Final centering with drone geometry compensation")

        if not center_on_bullseye_precise(vehicle, detector, video_recorder):
            logging.warning("⚠️ Could not center perfectly - landing anyway")

        # Step 5: Land
        logging.info("⬇️ Step 5: Initiating landing at middle detection point")

        if set_mode(vehicle, "LAND"):
            logging.info("✅ Successfully switched to LAND mode")
            monitor_landing(vehicle)
            return True
        else:
            logging.error("❌ Failed to switch to LAND mode")
            return False

    except Exception as e:
        logging.error(f"Error in middle point landing: {str(e)}")
        return False

def center_on_bullseye_precise(vehicle, detector, video_recorder, max_attempts=6):
    """
    Precise centering with drone geometry compensation (15mm camera offset).
    """
    try:
        logging.info("🎯 Precise centering with 15mm camera offset compensation")

        # Drone geometry: camera is 15mm forward of flight controller center
        camera_offset_m = 0.015

        for attempt in range(max_attempts):
            # Get current frame
            frame = get_camera_frame(video_recorder)
            if frame is None:
                logging.warning(f"Attempt {attempt+1}: No camera frame")
                time.sleep(1)
                continue

            # Detect bullseye
            bullseyes, _ = detector.detect_bullseyes_in_frame(frame)
            if len(bullseyes) == 0:
                logging.warning(f"Attempt {attempt+1}: No bullseye detected")
                time.sleep(1)
                continue

            # Get best detection
            try:
                best_detection = max(bullseyes, key=lambda x: x[3])
                center_x, center_y, _, confidence = best_detection
            except Exception as e:
                logging.error(f"Error processing detection: {str(e)}")
                continue

            # Get frame dimensions
            try:
                frame_height, frame_width = frame.shape[:2]
                frame_center_x = frame_width // 2
                frame_center_y = frame_height // 2
            except Exception as e:
                logging.error(f"Error getting frame dimensions: {str(e)}")
                continue

            # Calculate offset from frame center
            try:
                offset_x = float(center_x) - float(frame_center_x)
                offset_y = float(center_y) - float(frame_center_y)
                offset_distance = (offset_x**2 + offset_y**2)**0.5
            except Exception as e:
                logging.error(f"Error calculating offset: {str(e)}")
                continue

            logging.info(f"Centering attempt {attempt+1}: Bullseye at ({center_x}, {center_y}), "
                        f"offset ({offset_x:.0f}, {offset_y:.0f})px, distance: {offset_distance:.1f}px")

            # Check if well centered
            if offset_distance <= 12:  # Tight tolerance
                logging.info("🎯 PERFECTLY CENTERED! Ready for landing")
                return True

            # Calculate correction with geometry compensation
            pixels_per_meter = 80  # Rough estimate at higher altitude

            # Convert to meters
            offset_x_m = offset_x / pixels_per_meter
            offset_y_m = offset_y / pixels_per_meter

            # Apply 15mm camera offset compensation
            # When camera sees bullseye center, move drone so FC center aligns
            corrected_offset_y_m = offset_y_m - camera_offset_m

            # Gentle correction velocities
            max_velocity = 0.15  # Very gentle
            velocity_east = max(-max_velocity, min(max_velocity, offset_x_m * 0.3))
            velocity_north = max(-max_velocity, min(max_velocity, -corrected_offset_y_m * 0.3))

            logging.info(f"Precision correction: N={velocity_north:.3f}, E={velocity_east:.3f} m/s")

            # Apply gentle correction
            send_ned_velocity(vehicle, velocity_north, velocity_east, 0, 2)
            time.sleep(3)  # Wait for stabilization

        logging.info("🎯 Centering completed (may not be perfect)")
        return True

    except Exception as e:
        logging.error(f"Error in precise centering: {str(e)}")
        return True  # Continue anyway

def search_in_circle(vehicle, center_location, detector, video_recorder, radius=0.5, points=6):
    """
    Small circle search around center point.
    """
    try:
        logging.info(f"🔍 Small circle search: {radius}m radius, {points} points")

        center_lat, center_lon, center_alt = center_location

        for i in range(points):
            angle = (i * 360 / points) * math.pi / 180

            north_offset = radius * math.cos(angle)
            east_offset = radius * math.sin(angle)

            search_location = get_location_metres(center_location, north_offset, east_offset)
            search_lat, search_lon, search_alt = search_location

            logging.info(f"   Point {i+1}/{points}: {search_lat:.7f}, {search_lon:.7f}")

            command_waypoint(vehicle, search_lat, search_lon, center_alt)
            time.sleep(1.5)

            frame = get_camera_frame(video_recorder)
            if frame is not None:
                bullseyes, _ = detector.detect_bullseyes_in_frame(frame)
                if len(bullseyes) > 0:
                    logging.info(f"🎯 Found bullseye at circle point {i+1}!")
                    current_location = get_location(vehicle)
                    return current_location if current_location else search_location

        return None

    except Exception as e:
        logging.error(f"Error in circle search: {str(e)}")
        return None

def get_camera_frame(video_recorder):
    """Get frame from shared video recorder"""
    try:
        if video_recorder is not None and hasattr(video_recorder, 'cap') and video_recorder.cap is not None:
            ret, frame = video_recorder.cap.read()
            if ret:
                return frame
        return None
    except Exception as e:
        logging.warning(f"Error getting camera frame: {str(e)}")
        return None

def command_waypoint(vehicle, lat, lon, alt):
    """Send waypoint command"""
    try:
        vehicle.mav.set_position_target_global_int_send(
            0, vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        return True
    except Exception as e:
        logging.error(f"Error sending waypoint: {str(e)}")
        return False

def wait_for_waypoint_blocking(vehicle, target_lat, target_lon, target_altitude, timeout=45, tolerance=1.5):
    """Blocking wait for waypoint arrival"""
    if not vehicle:
        return False

    try:
        start_time = time.time()
        target_location = (target_lat, target_lon, 0)
        stable_count = 0
        required_stable_readings = 3

        while time.time() - start_time < timeout:
            current_location = get_location(vehicle)
            if current_location:
                distance = get_distance_metres(current_location, target_location)

                if distance <= tolerance:
                    stable_count += 1
                    if stable_count >= required_stable_readings:
                        logging.info(f"✅ Waypoint reached (distance: {distance:.2f}m)")
                        return True
                else:
                    stable_count = 0

            time.sleep(0.3)

        logging.warning("⏰ Waypoint timeout")
        return False

    except Exception as e:
        logging.error(f"Error waiting for waypoint: {str(e)}")
        return False

def wait_for_altitude_blocking(vehicle, target_altitude, timeout=30, tolerance=0.3):
    """Blocking wait for altitude"""
    try:
        start_time = time.time()
        stable_count = 0
        required_stable_readings = 3

        while time.time() - start_time < timeout:
            try:
                msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                if msg:
                    current_altitude = msg.relative_alt / 1000.0
                    altitude_diff = abs(current_altitude - target_altitude)

                    if altitude_diff <= tolerance:
                        stable_count += 1
                        if stable_count >= required_stable_readings:
                            logging.info(f"✅ Altitude reached: {current_altitude:.2f}m")
                            return True
                    else:
                        stable_count = 0
            except:
                pass

            time.sleep(0.2)

        logging.warning("⏰ Altitude timeout")
        return False

    except Exception as e:
        logging.error(f"Error waiting for altitude: {str(e)}")
        return False

def monitor_landing(vehicle):
    """Monitor landing process"""
    try:
        logging.info("📉 Monitoring landing...")
        start_time = time.time()

        while time.time() - start_time < 60:
            if not check_if_armed(vehicle):
                logging.info("✅ Landing complete - vehicle disarmed")
                return True

            try:
                msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                if msg:
                    alt = msg.relative_alt / 1000.0
                    if alt < 0.5:
                        logging.info("✅ Near ground - landing complete")
                        return True
            except:
                pass

            time.sleep(2)

        return True

    except Exception as e:
        logging.error(f"Error monitoring landing: {str(e)}")
        return True

def wait_for_landing(vehicle):
    """Wait for vehicle to land and disarm"""
    try:
        start_time = time.time()
        while time.time() - start_time < 120:
            if not check_if_armed(vehicle):
                logging.info("✅ Vehicle has landed and disarmed")
                break
            time.sleep(2)
    except:
        pass
