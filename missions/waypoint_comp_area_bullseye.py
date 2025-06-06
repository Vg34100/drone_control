# missions/waypoint_comp_area_bullseye.py - NEW MODULE
"""
Competition Area Bullseye Detection with Configurable Actions
------------------------------------------------------------
Systematic search within a defined 4-corner boundary area for bullseye targets.
Collects all bullseye detections during systematic coverage, then executes
configurable action (land, drop, deliver) at the center of all detections.
"""

import logging
import time
import json
import cv2
import math
import os
import signal
from datetime import datetime
from pymavlink import mavutil

from drone.connection import get_vehicle_state
from drone.navigation import (
    run_preflight_checks, set_mode, arm_and_takeoff, return_to_launch,
    check_if_armed, disarm_vehicle, get_location, send_ned_velocity,
    get_distance_metres, get_location_metres
)
from drone.servo import open_claw, close_claw
from detection.bullseye_detector import BullseyeDetector
from detection.camera import initialize_camera, capture_frame, close_camera

# Global variable for mission data (needed for signal handler)
mission_data = None

def mission_competition_area_bullseye(vehicle, altitude=8, model_path="models/best.pt",
                                    confidence=0.5, video_recorder=None, action='land'):
    """
    Execute competition-ready area bullseye detection mission within 4-corner boundary.

    Implements systematic approach:
    1. Initial reconnaissance of boundary area
    2. Systematic coverage with lawnmower pattern
    3. Calculate center of all bullseye detections
    4. Execute specified action at center point

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters
        model_path: Path to bullseye YOLO model
        confidence: Detection confidence threshold
        video_recorder: Existing video recorder (optional)
        action: Action to perform at bullseye center ('land', 'drop', 'deliver')

    Returns:
        True if mission completed successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    # Global collections for signal handler access
    global mission_data
    mission_data = {
        'all_bullseye_detections': [],
        'output_dir': None,
        'boundary_corners': [],
        'mission_start_time': time.time()
    }

    def signal_handler(signum, frame):
        """Handle Ctrl+C interruption and save data"""
        logging.warning("\n🛑 Mission interrupted by user (Ctrl+C)")
        logging.info("💾 Saving collected data before exit...")

        try:
            if mission_data['output_dir'] and mission_data['all_bullseye_detections']:
                # Save whatever data we have collected
                save_all_bullseye_detections(mission_data['all_bullseye_detections'], mission_data['output_dir'])
                save_mission_summary(mission_data['all_bullseye_detections'], mission_data['boundary_corners'], mission_data['output_dir'])
                logging.info(f"📂 Partial results saved to: {mission_data['output_dir']}")

            # Attempt emergency RTL
            if vehicle:
                logging.info("🏠 Attempting emergency return to launch...")
                try:
                    return_to_launch(vehicle)
                except:
                    logging.error("Emergency RTL failed")
        except Exception as save_error:
            logging.error(f"Error saving data during interruption: {str(save_error)}")

        logging.info("Mission terminated by user")
        exit(1)

    # Register signal handler for graceful shutdown
    import signal
    signal.signal(signal.SIGINT, signal_handler)

    try:
        logging.info("=== COMPETITION AREA BULLSEYE DETECTION MISSION ===")
        logging.info(f"Flight altitude: {altitude}m")
        logging.info(f"Model: {model_path}")
        logging.info(f"Confidence threshold: {confidence}")
        logging.info(f"Action at bullseye center: {action.upper()}")

        # Validate action parameter
        valid_actions = ['land', 'drop', 'deliver']
        if action not in valid_actions:
            logging.error(f"Invalid action '{action}'. Must be one of: {valid_actions}")
            return False

        # Define the 4-corner boundary area for competition
        boundary_corners = [
            # (35.3482380, -119.1051073),  # Northwest
            # (35.3481549, -119.1051114),  # Southwest
            # (35.3481462, -119.1046983),  # Southeast
            # (35.3482402, -119.1046970),  # Northeast

(35.3482337,    -119.1050604)
(35.3481571,    -119.1050617)
(35.3481582,    -119.1047727)
(35.3482369,    -119.1047741)
        ]

        mission_data['boundary_corners'] = boundary_corners

        logging.info("Competition area boundary corners:")
        for i, (lat, lon) in enumerate(boundary_corners):
            logging.info(f"  Corner {i+1}: {lat:.7f}, {lon:.7f}")

        # Initialize bullseye detector
        detector = BullseyeDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            imgsz=160
        )

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/comp_area_bullseye_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        mission_data['output_dir'] = output_dir

        logging.info(f"📂 Results will be saved to: {output_dir}")

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

        # PHASE 1: Initial Reconnaissance
        logging.info("\n🔍 PHASE 1: BOUNDARY RECONNAISSANCE")
        reconnaissance_detections = perform_boundary_reconnaissance_bullseye(
            vehicle, boundary_corners, altitude, detector, video_recorder, output_dir
        )

        if reconnaissance_detections:
            logging.info(f"📊 Reconnaissance found {len(reconnaissance_detections)} bullseye detections")
            mission_data['all_bullseye_detections'].extend(reconnaissance_detections)
        else:
            logging.info("📊 No bullseyes detected during reconnaissance")

        # PHASE 2: Systematic Coverage
        logging.info("\n🔍 PHASE 2: SYSTEMATIC LAWNMOWER COVERAGE")
        coverage_detections = perform_systematic_coverage_bullseye(
            vehicle, boundary_corners, altitude, detector, video_recorder, output_dir
        )

        if coverage_detections:
            logging.info(f"📊 Systematic coverage found {len(coverage_detections)} additional bullseye detections")
            mission_data['all_bullseye_detections'].extend(coverage_detections)

        # PHASE 3: Execute Action at Bullseye Center
        logging.info("\n🎯 PHASE 3: BULLSEYE ACTION EXECUTION")
        if mission_data['all_bullseye_detections']:
            # Calculate center of all bullseye detections
            center_point = calculate_bullseye_center_point(mission_data['all_bullseye_detections'])

            if center_point:
                logging.info(f"📍 Calculated bullseye center: {center_point[0]:.7f}, {center_point[1]:.7f}")

                # Execute action at center point
                success = execute_bullseye_action_at_center(
                    vehicle, center_point, detector, video_recorder, altitude, action
                )

                if success:
                    logging.info(f"🎉 MISSION SUCCESS: {action.upper()} action completed at bullseye center!")

                    # Save final results
                    save_all_bullseye_detections(mission_data['all_bullseye_detections'], output_dir)
                    save_mission_summary(mission_data['all_bullseye_detections'], boundary_corners, output_dir)

                    return True
                else:
                    logging.error(f"❌ {action.upper()} action failed")
            else:
                logging.error("❌ Could not calculate bullseye center point")
        else:
            logging.info("📊 No bullseyes detected during entire mission")

        # Return to launch if no detections or action failed
        logging.info("\n🏠 RETURN TO LAUNCH")
        return_to_launch(vehicle)
        wait_for_landing(vehicle)

        # Save results even if no action was performed
        save_all_bullseye_detections(mission_data['all_bullseye_detections'], output_dir)
        save_mission_summary(mission_data['all_bullseye_detections'], boundary_corners, output_dir)

        logging.info("🎉 COMPETITION AREA BULLSEYE MISSION COMPLETED")
        logging.info(f"📂 Results saved to: {output_dir}")
        return True

    except KeyboardInterrupt:
        # This should be caught by signal handler, but just in case
        logging.warning("Mission interrupted")
        return False
    except Exception as e:
        logging.error(f"Critical error during competition area bullseye mission: {str(e)}")
        try:
            # Save whatever data we have before emergency RTL
            if mission_data['output_dir'] and mission_data['all_bullseye_detections']:
                save_all_bullseye_detections(mission_data['all_bullseye_detections'], mission_data['output_dir'])
                save_mission_summary(mission_data['all_bullseye_detections'], mission_data['boundary_corners'], mission_data['output_dir'])
                logging.info(f"📂 Partial results saved to: {mission_data['output_dir']}")

            logging.info("Attempting emergency return to launch")
            return_to_launch(vehicle)
        except:
            pass
        return False

def perform_boundary_reconnaissance_bullseye(vehicle, boundary_corners, altitude, detector, video_recorder, output_dir):
    """
    Phase 1: Quick reconnaissance flight around boundary perimeter for bullseyes
    """
    try:
        logging.info("🚁 Flying boundary reconnaissance pattern for bullseyes")

        detections = []

        # Fly to each corner with bullseye detection
        for i, (corner_lat, corner_lon) in enumerate(boundary_corners):
            logging.info(f"📍 Reconnaissance point {i+1}/4: {corner_lat:.7f}, {corner_lon:.7f}")

            # Command waypoint
            command_waypoint(vehicle, corner_lat, corner_lon, altitude)

            # Wait for arrival while detecting bullseyes
            corner_detections = monitor_flight_with_bullseye_detection(
                vehicle, (corner_lat, corner_lon), altitude, detector, video_recorder, output_dir, timeout=45
            )

            detections.extend(corner_detections)

            if corner_detections:
                logging.info(f"🎯 Corner {i+1} complete: {len(corner_detections)} bullseye detections")
            else:
                logging.info(f"📍 Corner {i+1} complete: no bullseyes detected")

            # Brief pause for stability
            time.sleep(2)

        return detections

    except Exception as e:
        logging.error(f"Error during boundary reconnaissance: {str(e)}")
        return []

def perform_systematic_coverage_bullseye(vehicle, boundary_corners, altitude, detector, video_recorder, output_dir):
    """
    Phase 2: Systematic lawnmower pattern coverage within boundary for bullseyes
    """
    try:
        logging.info("🚁 Executing systematic lawnmower pattern for bullseye detection")

        # Calculate boundary box
        lats = [corner[0] for corner in boundary_corners]
        lons = [corner[1] for corner in boundary_corners]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Generate lawnmower pattern with appropriate spacing for bullseye detection
        pattern_spacing = 0.000040  # ~4.5m spacing for better bullseye coverage

        pattern_waypoints = generate_lawnmower_pattern(
            min_lat, max_lat, min_lon, max_lon, pattern_spacing
        )

        logging.info(f"📐 Generated {len(pattern_waypoints)} waypoints for systematic bullseye coverage")

        detections = []

        # Execute lawnmower pattern
        for i, (wp_lat, wp_lon) in enumerate(pattern_waypoints):
            logging.info(f"📍 Coverage waypoint {i+1}/{len(pattern_waypoints)}: {wp_lat:.7f}, {wp_lon:.7f}")

            # Command waypoint
            command_waypoint(vehicle, wp_lat, wp_lon, altitude)

            # Monitor flight with bullseye detection
            wp_detections = monitor_flight_with_bullseye_detection(
                vehicle, (wp_lat, wp_lon), altitude, detector, video_recorder, output_dir, timeout=30
            )

            detections.extend(wp_detections)

            if len(wp_detections) > 0:
                logging.info(f"🎯 Waypoint {i+1}: {len(wp_detections)} bullseye detections")

        return detections

    except Exception as e:
        logging.error(f"Error during systematic coverage: {str(e)}")
        return []

def monitor_flight_with_bullseye_detection(vehicle, target_location, altitude, detector, video_recorder, output_dir, timeout=30):
    """
    Monitor flight to waypoint while continuously detecting bullseyes and saving detection data
    """
    try:
        detections = []
        target_lat, target_lon = target_location
        target_pos = (target_lat, target_lon, 0)

        start_time = time.time()
        detection_interval = 0.15  # Check every 150ms for good coverage
        last_detection_check = 0

        while time.time() - start_time < timeout:
            current_time = time.time()

            # Periodic detection checks
            if current_time - last_detection_check >= detection_interval:
                current_location = get_location(vehicle)
                if current_location:
                    # Get camera frame
                    frame = get_camera_frame(video_recorder)
                    if frame is not None:
                        # Detect bullseyes
                        bullseyes, _ = detector.detect_bullseyes_in_frame(frame)

                        for bullseye_data in bullseyes:
                            center_x, center_y, bbox_info, confidence = bullseye_data

                            detection_dict = {
                                'gps_location': current_location,
                                'camera_center': (center_x, center_y),
                                'bbox': bbox_info['bbox'],
                                'confidence': confidence,
                                'timestamp': current_time
                            }

                            detections.append(detection_dict)

                            logging.info(f"🎯 BULLSEYE: GPS {current_location[0]:.7f}, {current_location[1]:.7f}, Conf: {confidence:.3f}")

                            # Save bullseye detection immediately with cropped image
                            save_bullseye_detection_immediate(detection_dict, frame, output_dir)

                    # Check if reached waypoint
                    distance = get_distance_metres(current_location, target_pos)
                    if distance <= 1.5:
                        logging.debug(f"✅ Reached waypoint (distance: {distance:.1f}m)")
                        break

                last_detection_check = current_time

            time.sleep(0.05)

        return detections

    except Exception as e:
        logging.error(f"Error monitoring flight with bullseye detection: {str(e)}")
        return []

def calculate_bullseye_center_point(all_detections):
    """
    Calculate the center GPS point of all bullseye detections.
    Returns (lat, lon, alt) of the center point.
    """
    try:
        if len(all_detections) == 0:
            return None

        # Extract all GPS coordinates
        latitudes = [d['gps_location'][0] for d in all_detections]
        longitudes = [d['gps_location'][1] for d in all_detections]
        altitudes = [d['gps_location'][2] for d in all_detections]

        # Calculate center point (arithmetic mean)
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        center_alt = sum(altitudes) / len(altitudes)

        logging.info(f"📊 Bullseye detection spread:")
        logging.info(f"   Lat range: {min(latitudes):.7f} to {max(latitudes):.7f}")
        logging.info(f"   Lon range: {min(longitudes):.7f} to {max(longitudes):.7f}")
        logging.info(f"   Center point: {center_lat:.7f}, {center_lon:.7f}")

        return (center_lat, center_lon, center_alt)

    except Exception as e:
        logging.error(f"Error calculating bullseye center point: {str(e)}")
        return None

def execute_bullseye_action_at_center(vehicle, center_point, detector, video_recorder, original_altitude, action):
    """
    Execute the specified action at the calculated bullseye center point.

    Args:
        vehicle: The connected mavlink object
        center_point: (lat, lon, alt) of the center detection point
        detector: BullseyeDetector instance
        video_recorder: Video recorder instance
        original_altitude: Original flight altitude
        action: Action to perform ('land', 'drop', 'deliver')

    Returns:
        True if action completed successfully
    """
    try:
        logging.info(f"🎯 EXECUTING {action.upper()} ACTION AT BULLSEYE CENTER")

        center_lat, center_lon, center_alt = center_point

        # Step 1: Navigate to center point
        logging.info(f"📍 Step 1: Flying to bullseye center {center_lat:.7f}, {center_lon:.7f}")

        command_waypoint(vehicle, center_lat, center_lon, original_altitude)

        # BLOCKING wait for arrival
        if not wait_for_waypoint_blocking(vehicle, center_lat, center_lon, original_altitude, timeout=30, tolerance=1.5):
            logging.error("❌ Failed to reach bullseye center")
            return False

        logging.info("✅ Reached bullseye center location")

        # Step 2: Verify bullseye at center point
        frame = get_camera_frame(video_recorder)
        if frame is not None:
            bullseyes, _ = detector.detect_bullseyes_in_frame(frame)
            if len(bullseyes) == 0:
                logging.warning("⚠️ No bullseye at center point - searching in small circle")

                # Small circle search at center point
                found_location = search_bullseye_in_circle(vehicle, center_point, detector, video_recorder, radius=0.5)
                if found_location:
                    center_point = found_location
                    center_lat, center_lon, center_alt = center_point
                    logging.info("🎯 Found bullseye in circle search")
                else:
                    logging.warning("⚠️ No bullseye in circle - proceeding with original center point")

        # Step 3: Climb for better detection view
        detection_altitude = original_altitude + 2
        logging.info(f"📈 Step 3: Climbing to {detection_altitude}m for precision positioning")

        command_waypoint(vehicle, center_lat, center_lon, detection_altitude)

        if not wait_for_altitude_blocking(vehicle, detection_altitude, timeout=20, tolerance=0.3):
            logging.warning("⚠️ Climb timeout - continuing anyway")

        # Step 4: Center on bullseye with drone geometry compensation
        logging.info("🎯 Step 4: Final centering with drone geometry compensation")

        if not center_on_bullseye_precise(vehicle, detector, video_recorder):
            logging.warning("⚠️ Could not center perfectly - proceeding anyway")

        # Step 5: Execute specific action based on action parameter
        if action == 'land':
            return execute_land_action(vehicle)
        elif action == 'drop':
            return execute_drop_action(vehicle)
        elif action == 'deliver':
            return execute_deliver_action(vehicle, center_lat, center_lon)
        else:
            logging.error(f"❌ Unknown action: {action}")
            return False

    except Exception as e:
        logging.error(f"Error in bullseye action execution: {str(e)}")
        return False

def execute_land_action(vehicle):
    """
    Execute LAND action - land directly on the bullseye.
    """
    try:
        logging.info("🛬 EXECUTING LAND ACTION - Landing on bullseye")

        if set_mode(vehicle, "LAND"):
            logging.info("✅ Successfully switched to LAND mode")
            monitor_landing(vehicle)
            logging.info("🎯 LAND ACTION COMPLETED")
            return True
        else:
            logging.error("❌ Failed to switch to LAND mode")
            return False

    except Exception as e:
        logging.error(f"Error in land action: {str(e)}")
        return False

def execute_drop_action(vehicle):
    """
    Execute DROP action - drop payload at current altitude then RTL.
    """
    try:
        logging.info("📦 EXECUTING DROP ACTION - Dropping payload at altitude")

        # Step 1: Drop the payload using close_claw
        logging.info("🤏 Step 1: Releasing payload (close_claw)")
        close_claw_success = close_claw(vehicle)

        if close_claw_success:
            logging.info("✅ Payload released successfully")
        else:
            logging.warning("⚠️ Payload release may have failed - continuing")

        # Step 2: Brief hover to ensure payload clears
        logging.info("⏸️ Step 2: Brief hover to ensure payload clears")
        time.sleep(3)

        # Step 3: Return to launch
        logging.info("🏠 Step 3: Returning to launch")
        rtl_success = return_to_launch(vehicle)

        if rtl_success:
            wait_for_landing(vehicle)
            logging.info("🎯 DROP ACTION COMPLETED")
            return True
        else:
            logging.error("❌ RTL failed after drop")
            return False

    except Exception as e:
        logging.error(f"Error in drop action: {str(e)}")
        return False

def execute_deliver_action(vehicle, target_lat, target_lon):
    """
    Execute DELIVER action - descend to 1-2m above ground, drop payload, then RTL.
    """
    try:
        logging.info("🚚 EXECUTING DELIVER ACTION - Descending for precision delivery")

        # Step 1: Descend to delivery altitude (1.5m above ground)
        delivery_altitude = 1.5
        logging.info(f"⬇️ Step 1: Descending to delivery altitude ({delivery_altitude}m)")

        command_waypoint(vehicle, target_lat, target_lon, delivery_altitude)

        if not wait_for_altitude_blocking(vehicle, delivery_altitude, timeout=30, tolerance=0.2):
            logging.warning("⚠️ Failed to reach exact delivery altitude - proceeding anyway")

        # Step 2: Final hover for stability
        logging.info("⏸️ Step 2: Stabilizing at delivery altitude")
        time.sleep(2)

        # Step 3: Drop the payload using close_claw
        logging.info("🤏 Step 3: Releasing payload for precision delivery (close_claw)")
        close_claw_success = close_claw(vehicle)

        if close_claw_success:
            logging.info("✅ Payload delivered successfully")
        else:
            logging.warning("⚠️ Payload delivery may have failed - continuing")

        # Step 4: Brief hover to ensure payload clears and settles
        logging.info("⏸️ Step 4: Brief hover to ensure safe payload delivery")
        time.sleep(3)

        # Step 5: Climb back up slightly before RTL for safety
        safety_altitude = 5.0
        logging.info(f"📈 Step 5: Climbing to safety altitude ({safety_altitude}m) before RTL")

        command_waypoint(vehicle, target_lat, target_lon, safety_altitude)

        if not wait_for_altitude_blocking(vehicle, safety_altitude, timeout=20, tolerance=0.3):
            logging.warning("⚠️ Safety climb timeout - proceeding with RTL anyway")

        # Step 6: Return to launch
        logging.info("🏠 Step 6: Returning to launch")
        rtl_success = return_to_launch(vehicle)

        if rtl_success:
            wait_for_landing(vehicle)
            logging.info("🎯 DELIVER ACTION COMPLETED")
            return True
        else:
            logging.error("❌ RTL failed after delivery")
            return False

    except Exception as e:
        logging.error(f"Error in deliver action: {str(e)}")
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
                logging.info("🎯 PERFECTLY CENTERED! Ready for action")
                return True

            # Calculate correction with geometry compensation
            pixels_per_meter = 80  # Rough estimate at higher altitude

            # Convert to meters
            offset_x_m = offset_x / pixels_per_meter
            offset_y_m = offset_y / pixels_per_meter

            # Apply 15mm camera offset compensation
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

def search_bullseye_in_circle(vehicle, center_location, detector, video_recorder, radius=0.5, points=6):
    """
    Small circle search around center point for bullseye.
    """
    try:
        logging.info(f"🔍 Small circle search for bullseye: {radius}m radius, {points} points")

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

def save_bullseye_detection_immediate(detection_dict, frame, output_dir):
    """
    Immediately save bullseye detection with cropped image and metadata
    """
    try:
        # Generate unique timestamp and ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Count existing bullseye files to get next ID
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('bullseye_') and f.endswith('.jpg')]
        bullseye_id = len(existing_files) + 1

        # Validate detection data
        if 'bbox' not in detection_dict or 'gps_location' not in detection_dict:
            logging.error("Invalid detection data for saving bullseye")
            return False

        # Extract crop coordinates with validation
        x1, y1, x2, y2 = detection_dict['bbox']
        h, w = frame.shape[:2]

        # Validate and fix bbox if needed
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            logging.warning(f"Invalid bbox {detection_dict['bbox']}, using center crop fallback")
            center_x, center_y = detection_dict.get('camera_center', (w//2, h//2))
            crop_size = 50
            x1 = max(0, center_x - crop_size)
            y1 = max(0, center_y - crop_size)
            x2 = min(w, center_x + crop_size)
            y2 = min(h, center_y + crop_size)

        # Add padding to crop
        padding = 30  # Increased padding for better context
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(w, x2 + padding)
        y2_crop = min(h, y2 + padding)

        # Extract crop
        cropped_image = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        if cropped_image.size == 0:
            logging.error("Cropped image is empty, using full frame")
            cropped_image = frame

        # Save cropped image
        crop_filename = f"bullseye_{bullseye_id:03d}_{timestamp}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save image with high quality
        success = cv2.imwrite(crop_path, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not success:
            logging.error(f"Failed to save cropped image: {crop_filename}")
            return False

        # Create metadata file alongside image
        metadata_filename = f"bullseye_{bullseye_id:03d}_{timestamp}.json"
        metadata_path = os.path.join(output_dir, metadata_filename)

        gps_location = detection_dict['gps_location']
        bullseye_metadata = {
            'bullseye_id': bullseye_id,
            'timestamp': detection_dict.get('timestamp', time.time()),
            'timestamp_str': datetime.fromtimestamp(detection_dict.get('timestamp', time.time())).isoformat(),
            'gps_location': {
                'latitude': gps_location[0],
                'longitude': gps_location[1],
                'altitude': gps_location[2]
            },
            'detection_info': {
                'confidence': detection_dict.get('confidence', 0.0),
                'camera_center': detection_dict.get('camera_center', (0, 0)),
                'bbox_original': detection_dict.get('bbox', (0, 0, 0, 0)),
                'bbox_cropped': (x1_crop, y1_crop, x2_crop, y2_crop)
            },
            'image_files': {
                'cropped_image': crop_filename,
                'image_dimensions': {
                    'width': cropped_image.shape[1],
                    'height': cropped_image.shape[0]
                }
            }
        }

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(bullseye_metadata, f, indent=2)

        logging.info(f"💾 Saved bullseye #{bullseye_id}: {crop_filename}")
        logging.info(f"📍 GPS: {gps_location[0]:.7f}, {gps_location[1]:.7f} | Conf: {detection_dict.get('confidence', 0.0):.3f}")

        return True

    except Exception as e:
        logging.error(f"Error saving bullseye detection: {str(e)}")
        return False

def save_all_bullseye_detections(all_detections, output_dir):
    """
    Save comprehensive JSON summary of all bullseye detections found during mission
    """
    try:
        if not all_detections:
            logging.info("No bullseye detections found to save")
            return 0

        # Create comprehensive results structure
        bullseye_summary = []

        for i, detection in enumerate(all_detections, 1):
            gps_location = detection.get('gps_location', (0, 0, 0))

            bullseye_entry = {
                'detection_order': i,
                'gps_location': {
                    'latitude': gps_location[0],
                    'longitude': gps_location[1],
                    'altitude': gps_location[2]
                },
                'detection_details': {
                    'confidence': detection.get('confidence', 0.0),
                    'camera_center': detection.get('camera_center', (0, 0)),
                    'bbox': detection.get('bbox', (0, 0, 0, 0)),
                    'timestamp': detection.get('timestamp', time.time()),
                    'timestamp_str': datetime.fromtimestamp(detection.get('timestamp', time.time())).isoformat()
                }
            }

            bullseye_summary.append(bullseye_entry)

        # Create final JSON structure
        final_results = {
            'mission_info': {
                'mission_type': 'competition_area_bullseye',
                'mission_timestamp': datetime.now().isoformat(),
                'total_bullseyes_found': len(all_detections),
                'mission_duration_seconds': time.time() - (all_detections[0].get('timestamp', time.time()) if all_detections else time.time())
            },
            'summary_statistics': {
                'total_detections': len(all_detections),
                'average_confidence': sum(d.get('confidence', 0) for d in all_detections) / len(all_detections),
                'confidence_range': {
                    'min': min(d.get('confidence', 0) for d in all_detections),
                    'max': max(d.get('confidence', 0) for d in all_detections)
                }
            },
            'bullseye_detections': bullseye_summary
        }

        # Save comprehensive results
        json_filename = "bullseye_detections_results.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Also save a simple coordinates-only file for quick reference
        simple_coords = [
            {
                'id': i,
                'latitude': bullseye['gps_location']['latitude'],
                'longitude': bullseye['gps_location']['longitude'],
                'confidence': bullseye['detection_details']['confidence']
            }
            for i, bullseye in enumerate(bullseye_summary, 1)
        ]

        simple_filename = "bullseye_coordinates_only.json"
        simple_path = os.path.join(output_dir, simple_filename)

        with open(simple_path, 'w') as f:
            json.dump({
                'total_bullseyes': len(simple_coords),
                'coordinates': simple_coords
            }, f, indent=2)

        logging.info(f"💾 Saved {len(all_detections)} bullseye detections to:")
        logging.info(f"  📄 Comprehensive: {json_filename}")
        logging.info(f"  📄 Simple: {simple_filename}")

        return len(all_detections)

    except Exception as e:
        logging.error(f"Error saving bullseye detections summary: {str(e)}")
        return 0

def save_mission_summary(all_detections, boundary_corners, output_dir):
    """
    Save comprehensive mission summary
    """
    try:
        summary = {
            'mission_info': {
                'mission_type': 'competition_area_bullseye',
                'timestamp': datetime.now().isoformat(),
                'boundary_area': {
                    'corners': [{'lat': corner[0], 'lon': corner[1]} for corner in boundary_corners]
                }
            },
            'detection_summary': {
                'total_bullseye_detections': len(all_detections),
                'detection_density': len(all_detections) / len(boundary_corners) if boundary_corners else 0
            },
            'all_detections': [
                {
                    'gps_location': d.get('gps_location', (0, 0, 0)),
                    'confidence': d.get('confidence', 0.0),
                    'timestamp': d.get('timestamp', 0)
                } for d in all_detections
            ]
        }

        summary_filename = "mission_summary.json"
        summary_path = os.path.join(output_dir, summary_filename)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logging.info(f"💾 Saved mission summary to {summary_filename}")

    except Exception as e:
        logging.error(f"Error saving mission summary: {str(e)}")

def generate_lawnmower_pattern(min_lat, max_lat, min_lon, max_lon, spacing):
    """
    Generate lawnmower pattern waypoints within the boundary
    """
    try:
        waypoints = []

        # Create north-south passes
        current_lon = min_lon
        going_north = True

        while current_lon <= max_lon:
            if going_north:
                # South to north pass
                current_lat = min_lat
                while current_lat <= max_lat:
                    waypoints.append((current_lat, current_lon))
                    current_lat += spacing
            else:
                # North to south pass
                current_lat = max_lat
                while current_lat >= min_lat:
                    waypoints.append((current_lat, current_lon))
                    current_lat -= spacing

            going_north = not going_north
            current_lon += spacing

        return waypoints

    except Exception as e:
        logging.error(f"Error generating lawnmower pattern: {str(e)}")
        return []

# Helper functions (reused from other modules)
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
        if not vehicle:
            return False

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
                        return True
                else:
                    stable_count = 0

            time.sleep(0.3)

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
