# missions/waypoint_gcp.py - UPDATED WITH MINIMAL CHANGES
"""
GCP Detection with Marker Search and Numbered Marker Collection - Updated
------------------------------------------------------------------------
Added early termination and configurable thresholds while keeping everything else the same.
"""

import logging
import time
import json
import cv2
import math
import os
from datetime import datetime
from pymavlink import mavutil

from drone.connection import get_vehicle_state
from drone.navigation import (
    run_preflight_checks, set_mode, arm_and_takeoff, return_to_launch,
    check_if_armed, disarm_vehicle, get_location, send_ned_velocity,
    get_distance_metres, get_location_metres
)
from detection.camera import initialize_camera, capture_frame, close_camera

# Try to import YOLO, fallback if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available - install ultralytics: pip install ultralytics")

class GCPDetector:
    """GCP detector for markers and numbered markers using YOLO"""

    def __init__(self, model_path="models/best-gcp.pt", confidence_threshold=0.5, imgsz=160):
        """Initialize GCP detector with YOLO model"""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")

        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.imgsz = imgsz

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GCP model file not found: {model_path}")

        logging.info(f"Loading GCP YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.detections_count = 0

    def detect_gcp_markers_in_frame(self, frame):
        """
        Detect GCP markers in frame using YOLO model.

        Returns:
            List of detections: [(class_name, center_x, center_y, bbox, confidence), ...]
        """
        if frame is None:
            return [], frame

        try:
            start_time = time.time()

            # Run YOLO inference
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False
            )

            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1

            # Process results
            detections = []
            debug_image = frame.copy()

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Get class name
                        class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"

                        # Calculate center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Create bbox info
                        bbox = (x1, y1, x2, y2)

                        # Add to detections
                        detections.append((class_name, center_x, center_y, bbox, confidence))
                        self.detections_count += 1

                        # Draw on debug image
                        self._draw_detection(debug_image, x1, y1, x2, y2, center_x, center_y,
                                           class_name, confidence)

            # Add frame info
            self._add_frame_info(debug_image, len(detections), inference_time)

            return detections, debug_image

        except Exception as e:
            logging.error(f"Error in GCP detection: {str(e)}")
            return [], frame

    def _draw_detection(self, image, x1, y1, x2, y2, center_x, center_y, class_name, confidence):
        """Draw detection on image"""
        # Color coding: green for markers, blue for marker-numbered
        color = (0, 255, 0) if class_name == 'markers' else (255, 0, 0)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cv2.circle(image, (center_x, center_y), 5, color, -1)

        # Draw crosshair at center
        cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), color, 2)
        cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), color, 2)

        # Add label
        label = f"{class_name}: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _add_frame_info(self, image, num_detections, inference_time):
        """Add frame information overlay"""
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Add frame info
        info_text = f"GCP FPS: {fps:.1f} | Detections: {num_detections} | Frame: {self.frame_count}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Add crosshair at frame center
        h, w = image.shape[:2]
        cv2.line(image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
        cv2.line(image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)
        cv2.circle(image, (w//2, h//2), 3, (0, 255, 255), -1)

def mission_waypoint_gcp_detection(vehicle, altitude=6, model_path="models/best-gcp.pt",
                                 confidence=0.5, loops=1, video_recorder=None,
                                 min_numbered_markers=3, min_general_markers=3,
                                 marker_distance_threshold=2.0):
    """
    Execute waypoint mission searching for GCP markers and numbered markers.

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters
        model_path: Path to GCP YOLO model
        confidence: Detection confidence threshold
        loops: Number of times to repeat waypoint pattern
        video_recorder: Existing video recorder (optional)
        min_numbered_markers: Minimum numbered markers to find before concluding (default: 3)
        min_general_markers: Minimum general markers to investigate before concluding (default: 3)
        marker_distance_threshold: Minimum distance between markers to consider them separate (default: 2.0m)

    Returns:
        True if mission completed successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("=== GCP MARKER DETECTION AND COLLECTION MISSION ===")
        logging.info(f"Flight altitude: {altitude}m")
        logging.info(f"Model: {model_path}")
        logging.info(f"Confidence threshold: {confidence}")
        logging.info(f"Min numbered markers for conclusion: {min_numbered_markers}")
        logging.info(f"Min general markers for conclusion: {min_general_markers}")
        logging.info(f"Marker distance threshold: {marker_distance_threshold}m")

        # Your specific waypoints
        waypoints = [
            (35.3481828, -119.1049256),  # Point 1
            (35.3481833, -119.1046789),  # Point 2
        ] * loops

        # Initialize GCP detector
        detector = GCPDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            imgsz=160
        )

        # Initialize collections
        numbered_markers = []  # For marker-numbered detections
        markers_to_investigate = []  # For markers that need circle search

        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/gcp_mission_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

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

        # UPDATED: Fly waypoints with early termination on marker detection
        for i, (target_lat, target_lon) in enumerate(waypoints):
            try:
                # Validate waypoint format
                if not waypoints or len(waypoints[i]) < 2:
                    logging.error(f"Invalid waypoint {i+1}: {waypoints[i]}")
                    continue

                target_lat, target_lon = waypoints[i][0], waypoints[i][1]

                # Validate coordinates
                if not (-90 <= target_lat <= 90) or not (-180 <= target_lon <= 180):
                    logging.error(f"Invalid coordinates for waypoint {i+1}: {target_lat}, {target_lon}")
                    continue

                logging.info(f"\n📍 Flying to waypoint {i+1}/{len(waypoints)}: {target_lat:.7f}, {target_lon:.7f}")

                # Send waypoint command with error checking
                if not command_waypoint(vehicle, target_lat, target_lon, altitude):
                    logging.error(f"Failed to send waypoint {i+1} command, continuing to next waypoint")
                    continue

                # Monitor flight path and collect detections with EARLY TERMINATION
                try:
                    path_detections, should_terminate = collect_gcp_detections_with_termination(
                        vehicle, target_lat, target_lon, altitude, detector,
                        video_recorder, numbered_markers, markers_to_investigate, output_dir,
                        marker_distance_threshold
                    )

                    # Count detections safely
                    numbered_count = sum(1 for d in path_detections if d.get('class') == 'marker-numbered')
                    markers_count = sum(1 for d in path_detections if d.get('class') == 'markers')

                    logging.info(f"📊 Path {i+1} summary: {numbered_count} numbered, {markers_count} markers")

                    # EARLY TERMINATION: If markers detected, stop waypoint sequence and investigate
                    if should_terminate:
                        logging.info(f"🛑 EARLY TERMINATION: Markers detected, starting immediate investigation")
                        break

                except Exception as path_error:
                    logging.error(f"Error during path {i+1} detection collection: {str(path_error)}")
                    continue

            except Exception as waypoint_error:
                logging.error(f"Error processing waypoint {i+1}: {str(waypoint_error)}")
                continue

        # Process any markers that need circle investigation
        if markers_to_investigate:
            logging.info(f"\n🔍 INVESTIGATING {len(markers_to_investigate)} MARKERS WITH CIRCLE SEARCH")

            for marker_idx, marker_location in enumerate(markers_to_investigate):
                try:
                    # Validate marker location
                    if not marker_location or len(marker_location) < 2:
                        logging.error(f"Invalid marker location {marker_idx + 1}: {marker_location}")
                        continue

                    logging.info(f"\n🎯 Investigating marker {marker_idx + 1}/{len(markers_to_investigate)}")
                    logging.info(f"   Location: {marker_location[0]:.7f}, {marker_location[1]:.7f}")

                    # Go to marker location
                    if not command_waypoint(vehicle, marker_location[0], marker_location[1], altitude):
                        logging.error(f"Failed to command waypoint for marker {marker_idx + 1}")
                        continue

                    if not wait_for_waypoint_blocking(vehicle, marker_location[0], marker_location[1],
                                                    altitude, timeout=45, tolerance=1.5):
                        logging.warning(f"Failed to reach marker {marker_idx + 1}, skipping circle search")
                        continue

                    # Perform 1m circle search around marker with termination conditions
                    try:
                        circle_detections, investigation_complete = perform_circle_search(
                            vehicle, marker_location, detector, video_recorder,
                            numbered_markers, markers_to_investigate, output_dir,
                            radius=1.0, min_numbered_markers=min_numbered_markers,
                            min_general_markers=min_general_markers,
                            marker_distance_threshold=marker_distance_threshold
                        )

                        logging.info(f"🔍 Circle search complete: {len(circle_detections)} new detections")

                        # Check if investigation is complete
                        if investigation_complete:
                            logging.info(f"🎯 Investigation complete - found sufficient markers")
                            break

                    except Exception as circle_error:
                        logging.error(f"Error during circle search for marker {marker_idx + 1}: {str(circle_error)}")
                        continue

                except Exception as marker_error:
                    logging.error(f"Error investigating marker {marker_idx + 1}: {str(marker_error)}")
                    continue
        else:
            logging.info("🔍 No markers found that require circle investigation")

        # Save numbered markers to JSON
        try:
            save_numbered_markers_json(numbered_markers, output_dir)
        except Exception as save_error:
            logging.error(f"Error saving numbered markers JSON: {str(save_error)}")

        # Mission summary
        logging.info(f"\n📊 MISSION SUMMARY:")
        logging.info(f"   Numbered markers found: {len(numbered_markers)}")
        logging.info(f"   Markers investigated: {len(markers_to_investigate)}")
        logging.info(f"   Results saved to: {output_dir}")

        # Return to launch
        logging.info("\n🏠 RETURN TO LAUNCH")
        return_to_launch(vehicle)
        wait_for_landing(vehicle)

        logging.info("🎉 GCP DETECTION MISSION COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        logging.error(f"Critical error during GCP detection mission: {str(e)}")
        logging.exception("Full exception details:")
        try:
            logging.info("Attempting emergency return to launch")
            return_to_launch(vehicle)
        except Exception as rtl_error:
            logging.error(f"Emergency RTL also failed: {str(rtl_error)}")
        return False

def collect_gcp_detections_with_termination(vehicle, target_lat, target_lon, altitude, detector,
                                           video_recorder, numbered_markers, markers_to_investigate,
                                           output_dir, marker_distance_threshold=2.0):
    """
    Collect GCP detections during flight to waypoint with early termination capability.

    Returns:
        (detections, should_terminate): Tuple of detections list and termination flag
    """
    try:
        detections = []
        target_location = (target_lat, target_lon, 0)
        detection_check_interval = 0.15
        last_check = 0
        should_terminate = False

        # Add timeout to prevent infinite loops
        flight_start_time = time.time()
        max_flight_time = 120

        logging.info("🔍 Scanning for GCP markers during flight...")

        while True:
            current_time = time.time()

            # Safety timeout check
            if current_time - flight_start_time > max_flight_time:
                logging.warning(f"Flight to waypoint timed out after {max_flight_time}s")
                break

            if current_time - last_check >= detection_check_interval:
                # Get current GPS location
                try:
                    current_location = get_location(vehicle)
                    if not current_location:
                        time.sleep(0.05)
                        continue
                except Exception as loc_error:
                    logging.warning(f"Error getting location: {str(loc_error)}")
                    time.sleep(0.05)
                    continue

                # Get camera frame
                try:
                    frame = get_camera_frame(video_recorder)
                    if frame is None:
                        time.sleep(0.05)
                        continue
                except Exception as frame_error:
                    logging.warning(f"Error getting camera frame: {str(frame_error)}")
                    time.sleep(0.05)
                    continue

                # Detect GCP markers
                try:
                    gcp_detections, debug_image = detector.detect_gcp_markers_in_frame(frame)
                except Exception as detection_error:
                    logging.warning(f"Error during GCP detection: {str(detection_error)}")
                    time.sleep(0.05)
                    continue

                # Process each detection
                for detection_data in gcp_detections:
                    try:
                        if len(detection_data) < 5:
                            logging.warning(f"Invalid detection data format: {detection_data}")
                            continue

                        class_name, center_x, center_y, bbox, confidence = detection_data

                        detection_dict = {
                            'class': class_name,
                            'gps_location': current_location,
                            'camera_center': (center_x, center_y),
                            'bbox': bbox,
                            'confidence': confidence,
                            'timestamp': current_time
                        }

                        detections.append(detection_dict)

                        logging.info(f"🎯 {class_name.upper()} detected: GPS {current_location[0]:.7f}, {current_location[1]:.7f} | "
                                   f"Conf: {confidence:.3f}")

                        # Handle marker-numbered class
                        if class_name == 'marker-numbered':
                            try:
                                save_numbered_marker(detection_dict, frame, output_dir, numbered_markers)
                            except Exception as save_error:
                                logging.error(f"Error saving numbered marker: {str(save_error)}")

                        # Handle markers class - SET TERMINATION FLAG
                        elif class_name == 'markers':
                            try:
                                # Add to investigation list if not already close to an existing marker
                                should_add = True
                                for existing_marker in markers_to_investigate:
                                    if len(existing_marker) >= 2:
                                        distance = get_distance_metres(current_location, existing_marker)
                                        if distance < marker_distance_threshold:
                                            should_add = False
                                            break

                                if should_add:
                                    markers_to_investigate.append(current_location)
                                    logging.info(f"➕ Added marker for investigation: {current_location[0]:.7f}, {current_location[1]:.7f}")

                                # SET TERMINATION FLAG when any general marker is detected
                                should_terminate = True
                                logging.info(f"🛑 General marker detected - will terminate waypoint sequence after reaching this waypoint")

                            except Exception as marker_error:
                                logging.error(f"Error processing marker for investigation: {str(marker_error)}")

                    except Exception as process_error:
                        logging.error(f"Error processing individual detection: {str(process_error)}")
                        continue

                # Check if reached waypoint
                try:
                    distance = get_distance_metres(current_location, target_location)
                    if distance <= 1.5:
                        logging.info(f"✅ Reached waypoint (distance: {distance:.1f}m)")
                        break
                except Exception as distance_error:
                    logging.warning(f"Error calculating distance to waypoint: {str(distance_error)}")

                last_check = current_time

            time.sleep(0.02)

        return detections, should_terminate

    except Exception as e:
        logging.error(f"Critical error during GCP detection collection: {str(e)}")
        return [], False

def perform_circle_search(vehicle, center_location, detector, video_recorder,
                         numbered_markers, markers_to_investigate, output_dir,
                         radius=1.0, points=8, min_numbered_markers=3, min_general_markers=3,
                         marker_distance_threshold=2.0):
    """
    Perform circle search around a marker location with termination conditions.

    Returns:
        (circle_detections, investigation_complete): Tuple of detections and completion flag
    """
    try:
        # Validate input parameters
        if not center_location or len(center_location) < 3:
            logging.error(f"Invalid center location: {center_location}")
            return [], False

        if radius <= 0 or points <= 0:
            logging.error(f"Invalid circle parameters: radius={radius}, points={points}")
            return [], False

        logging.info(f"🔄 Performing {radius}m circle search with {points} points")
        logging.info(f"Will conclude if find {min_numbered_markers} numbered or {min_general_markers} general markers")

        center_lat, center_lon, center_alt = center_location
        circle_detections = []

        # Track unique markers found
        unique_numbered_markers = []
        unique_general_markers = []

        for i in range(points):
            try:
                angle = (i * 360 / points) * math.pi / 180

                north_offset = radius * math.cos(angle)
                east_offset = radius * math.sin(angle)

                search_location = get_location_metres(center_location, north_offset, east_offset)
                if not search_location or len(search_location) < 3:
                    logging.warning(f"Failed to calculate search location for point {i+1}")
                    continue

                search_lat, search_lon, search_alt = search_location

                logging.info(f"   🎯 Circle point {i+1}/{points}: {search_lat:.7f}, {search_lon:.7f}")

                # Move to circle point
                if not command_waypoint(vehicle, search_lat, search_lon, center_alt):
                    logging.warning(f"Failed to command circle point {i+1}")
                    continue

                # Wait briefly for movement
                time.sleep(2)

                # Check for detections at this point
                try:
                    frame = get_camera_frame(video_recorder)
                    if frame is not None:
                        gcp_detections, _ = detector.detect_gcp_markers_in_frame(frame)

                        if gcp_detections:
                            current_location = get_location(vehicle)
                            if current_location:
                                for detection_data in gcp_detections:
                                    try:
                                        if len(detection_data) < 5:
                                            continue

                                        class_name, center_x, center_y, bbox, confidence = detection_data

                                        detection_dict = {
                                            'class': class_name,
                                            'gps_location': current_location,
                                            'camera_center': (center_x, center_y),
                                            'bbox': bbox,
                                            'confidence': confidence,
                                            'timestamp': time.time(),
                                            'found_in_circle': True
                                        }

                                        circle_detections.append(detection_dict)

                                        logging.info(f"🎯 Circle detection: {class_name} at {current_location[0]:.7f}, {current_location[1]:.7f}")

                                        # Check if this is a new unique marker
                                        is_new_marker = True
                                        if class_name == 'marker-numbered':
                                            for existing_loc in unique_numbered_markers:
                                                if get_distance_metres(current_location, existing_loc) < marker_distance_threshold:
                                                    is_new_marker = False
                                                    break
                                            if is_new_marker:
                                                unique_numbered_markers.append(current_location)
                                                save_numbered_marker(detection_dict, frame, output_dir, numbered_markers)
                                        elif class_name == 'markers':
                                            for existing_loc in unique_general_markers:
                                                if get_distance_metres(current_location, existing_loc) < marker_distance_threshold:
                                                    is_new_marker = False
                                                    break
                                            if is_new_marker:
                                                unique_general_markers.append(current_location)
                                                markers_to_investigate.append(current_location)
                                                logging.info(f"➕ New marker found in circle, added for investigation")

                                        # CHECK TERMINATION CONDITIONS
                                        if len(unique_numbered_markers) >= min_numbered_markers:
                                            logging.info(f"🎯 Found {len(unique_numbered_markers)} numbered markers - concluding circle search")
                                            return circle_detections, True
                                        elif len(unique_general_markers) >= min_general_markers:
                                            logging.info(f"📍 Investigated {len(unique_general_markers)} general markers - concluding circle search")
                                            return circle_detections, True

                                    except Exception as det_error:
                                        logging.error(f"Error processing circle detection: {str(det_error)}")
                                        continue
                except Exception as frame_error:
                    logging.warning(f"Error processing frame at circle point {i+1}: {str(frame_error)}")

            except Exception as point_error:
                logging.error(f"Error at circle point {i+1}: {str(point_error)}")
                continue

        # Return to center after circle
        try:
            command_waypoint(vehicle, center_lat, center_lon, center_alt)
            time.sleep(2)
        except Exception as return_error:
            logging.warning(f"Error returning to circle center: {str(return_error)}")

        return circle_detections, False

    except Exception as e:
        logging.error(f"Critical error during circle search: {str(e)}")
        return [], False

def save_numbered_marker(detection_data, frame, output_dir, numbered_markers):
    """
    Save numbered marker detection to JSON and crop image.
    """
    try:
        # Validate inputs
        if not detection_data or 'bbox' not in detection_data:
            logging.error("Invalid detection data for saving numbered marker")
            return

        if frame is None:
            logging.error("No frame provided for saving numbered marker")
            return

        marker_id = len(numbered_markers) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Crop the detected region
        try:
            x1, y1, x2, y2 = detection_data['bbox']

            # Validate bbox coordinates
            h, w = frame.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                logging.warning(f"Invalid bbox coordinates: {detection_data['bbox']}, frame size: {w}x{h}")
                # Use center crop as fallback
                center_x, center_y = detection_data.get('camera_center', (w//2, h//2))
                crop_size = 50
                x1 = max(0, center_x - crop_size)
                y1 = max(0, center_y - crop_size)
                x2 = min(w, center_x + crop_size)
                y2 = min(h, center_y + crop_size)

            # Add some padding to the crop
            padding = 20
            x1_crop = max(0, x1 - padding)
            y1_crop = max(0, y1 - padding)
            x2_crop = min(w, x2 + padding)
            y2_crop = min(h, y2 + padding)

            cropped_image = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            if cropped_image.size == 0:
                logging.error("Cropped image is empty, using full frame")
                cropped_image = frame

        except Exception as crop_error:
            logging.error(f"Error cropping image: {str(crop_error)}, using full frame")
            cropped_image = frame

        # Save cropped image
        try:
            crop_filename = f"numbered_marker_{marker_id:03d}_{timestamp}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_image)
        except Exception as save_error:
            logging.error(f"Error saving cropped image: {str(save_error)}")
            crop_filename = "save_failed.jpg"

        # Prepare data for JSON
        try:
            gps_location = detection_data.get('gps_location', (0, 0, 0))
            camera_center = detection_data.get('camera_center', (0, 0))
            bbox = detection_data.get('bbox', (0, 0, 0, 0))
            confidence = detection_data.get('confidence', 0.0)
            timestamp_val = detection_data.get('timestamp', time.time())

            marker_data = {
                'id': marker_id,
                'gps_location': {
                    'latitude': gps_location[0],
                    'longitude': gps_location[1],
                    'altitude': gps_location[2]
                },
                'detection': {
                    'camera_center': camera_center,
                    'bbox': bbox,
                    'confidence': confidence
                },
                'timestamp': timestamp_val,
                'timestamp_str': datetime.fromtimestamp(timestamp_val).isoformat(),
                'cropped_image_path': crop_filename,
                'found_in_circle': detection_data.get('found_in_circle', False)
            }

            numbered_markers.append(marker_data)

            logging.info(f"💾 Saved numbered marker #{marker_id}: {crop_filename}")

        except Exception as data_error:
            logging.error(f"Error preparing numbered marker data: {str(data_error)}")

    except Exception as e:
        logging.error(f"Critical error saving numbered marker: {str(e)}")

def save_numbered_markers_json(numbered_markers, output_dir):
    """
    Save all numbered markers to JSON file.
    """
    try:
        json_filename = "numbered_markers_results.json"
        json_path = os.path.join(output_dir, json_filename)

        mission_summary = {
            'mission_timestamp': datetime.now().isoformat(),
            'total_numbered_markers_found': len(numbered_markers),
            'markers': numbered_markers
        }

        with open(json_path, 'w') as f:
            json.dump(mission_summary, f, indent=2)

        logging.info(f"💾 Saved {len(numbered_markers)} numbered markers to {json_filename}")

    except Exception as e:
        logging.error(f"Error saving JSON file: {str(e)}")

def get_camera_frame(video_recorder):
    """Get frame from shared video recorder with error handling"""
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
    """Send waypoint command with error handling"""
    try:
        if not vehicle:
            logging.error("No vehicle connection for waypoint command")
            return False

        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logging.error(f"Invalid coordinates: lat={lat}, lon={lon}")
            return False

        if alt < 0:
            logging.warning(f"Negative altitude: {alt}, using absolute value")
            alt = abs(alt)

        vehicle.mav.set_position_target_global_int_send(
            0, vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        return True
    except Exception as e:
        logging.error(f"Error sending waypoint command: {str(e)}")
        return False

def wait_for_waypoint_blocking(vehicle, target_lat, target_lon, target_altitude, timeout=45, tolerance=1.5):
    """Blocking wait for waypoint arrival with error handling"""
    if not vehicle:
        return False

    try:
        start_time = time.time()
        target_location = (target_lat, target_lon, 0)
        stable_count = 0
        required_stable_readings = 3

        # Add safety checks
        position_check_failures = 0
        max_position_failures = 5

        while time.time() - start_time < timeout:
            try:
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

                    position_check_failures = 0  # Reset failure count on success
                else:
                    position_check_failures += 1
                    if position_check_failures >= max_position_failures:
                        logging.error("Too many position check failures")
                        return False

            except Exception as check_error:
                logging.warning(f"Error during waypoint check: {str(check_error)}")
                position_check_failures += 1
                if position_check_failures >= max_position_failures:
                    logging.error("Too many waypoint check errors")
                    return False

            time.sleep(0.3)

        logging.warning("⏰ Waypoint timeout")
        return False

    except Exception as e:
        logging.error(f"Error waiting for waypoint: {str(e)}")
        return False

def wait_for_landing(vehicle):
    """Wait for vehicle to land and disarm with error handling"""
    try:
        start_time = time.time()
        timeout = 120  # 2 minutes timeout

        while time.time() - start_time < timeout:
            try:
                if not check_if_armed(vehicle):
                    logging.info("✅ Vehicle has landed and disarmed")
                    break
            except Exception as check_error:
                logging.warning(f"Error checking armed status: {str(check_error)}")

            time.sleep(2)

        if time.time() - start_time >= timeout:
            logging.warning("Landing wait timed out")

    except Exception as e:
        logging.error(f"Error waiting for landing: {str(e)}")
        # Don't fail the mission for landing wait errors
