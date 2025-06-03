# missions/waypoint_gcp.py - NEW GCP DETECTION AND COLLECTION MISSION
"""
GCP Detection with Marker Search and Numbered Marker Collection
--------------------------------------------------------------
Searches for GCP markers and numbered markers, performing circles around 'markers'
class detections and collecting 'marker-numbered' class detections to JSON.
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
                                 confidence=0.5, loops=1, video_recorder=None):
    """
    Execute waypoint mission searching for GCP markers and numbered markers.

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters
        model_path: Path to GCP YOLO model
        confidence: Detection confidence threshold
        loops: Number of times to repeat waypoint pattern
        video_recorder: Existing video recorder (optional)

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
        current_marker_index = 0

        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"gcp_mission_results_{timestamp}"
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

        # Fly waypoints and search for GCP markers
        for i, (target_lat, target_lon) in enumerate(waypoints, 1):
            logging.info(f"\n📍 Flying to waypoint {i}: {target_lat:.7f}, {target_lon:.7f}")

            # Send waypoint command
            command_waypoint(vehicle, target_lat, target_lon, altitude)

            # Monitor flight path and collect detections
            path_detections = collect_gcp_detections_during_flight(
                vehicle, target_lat, target_lon, altitude, detector,
                video_recorder, numbered_markers, markers_to_investigate, output_dir
            )

            logging.info(f"📊 Path {i} summary: {len([d for d in path_detections if d[0] == 'marker-numbered'])} numbered, "
                        f"{len([d for d in path_detections if d[0] == 'markers'])} markers")

        # Process any markers that need circle investigation
        if markers_to_investigate:
            logging.info(f"\n🔍 INVESTIGATING {len(markers_to_investigate)} MARKERS WITH CIRCLE SEARCH")

            for marker_idx, marker_location in enumerate(markers_to_investigate):
                logging.info(f"\n🎯 Investigating marker {marker_idx + 1}/{len(markers_to_investigate)}")
                logging.info(f"   Location: {marker_location[0]:.7f}, {marker_location[1]:.7f}")

                # Go to marker location
                command_waypoint(vehicle, marker_location[0], marker_location[1], altitude)

                if not wait_for_waypoint_blocking(vehicle, marker_location[0], marker_location[1],
                                                altitude, timeout=45, tolerance=1.5):
                    logging.warning(f"Failed to reach marker {marker_idx + 1}, skipping circle search")
                    continue

                # Perform 1m circle search around marker
                circle_detections = perform_circle_search(
                    vehicle, marker_location, detector, video_recorder,
                    numbered_markers, markers_to_investigate, output_dir, radius=1.0
                )

                logging.info(f"🔍 Circle search complete: {len(circle_detections)} new detections")

        # Save numbered markers to JSON
        save_numbered_markers_json(numbered_markers, output_dir)

        # Mission summary
        logging.info(f"\n📊 MISSION SUMMARY:")
        logging.info(f"   Numbered markers found: {len(numbered_markers)}")
        logging.info(f"   Markers investigated: {len(markers_to_investigate)}")
        logging.info(f"   Results saved to: {output_dir}")

        # Return to launch
        logging.info("\n🏠 RETURN TO LAUNCH")
        return_to_launch(vehicle)
        wait_for_landing(vehicle)

        logging.info("🎉 GCP DETECTION MISSION COMPLETED")
        return True

    except Exception as e:
        logging.error(f"Error during GCP detection mission: {str(e)}")
        try:
            return_to_launch(vehicle)
        except:
            pass
        return False

def collect_gcp_detections_during_flight(vehicle, target_lat, target_lon, altitude, detector,
                                        video_recorder, numbered_markers, markers_to_investigate, output_dir):
    """
    Collect GCP detections during flight to waypoint.
    """
    try:
        detections = []
        target_location = (target_lat, target_lon, 0)
        detection_check_interval = 0.15  # Check every 150ms
        last_check = 0

        logging.info("🔍 Scanning for GCP markers during flight...")

        while True:
            current_time = time.time()

            if current_time - last_check >= detection_check_interval:
                # Get current GPS location
                current_location = get_location(vehicle)
                if not current_location:
                    time.sleep(0.05)
                    continue

                # Get camera frame
                frame = get_camera_frame(video_recorder)
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Detect GCP markers
                gcp_detections, debug_image = detector.detect_gcp_markers_in_frame(frame)

                # Process each detection
                for class_name, center_x, center_y, bbox, confidence in gcp_detections:
                    detection_data = {
                        'class': class_name,
                        'gps_location': current_location,
                        'camera_center': (center_x, center_y),
                        'bbox': bbox,
                        'confidence': confidence,
                        'timestamp': current_time
                    }

                    detections.append(detection_data)

                    logging.info(f"🎯 {class_name.upper()} detected: GPS {current_location[0]:.7f}, {current_location[1]:.7f} | "
                               f"Conf: {confidence:.3f}")

                    # Handle marker-numbered class
                    if class_name == 'marker-numbered':
                        save_numbered_marker(detection_data, frame, output_dir, numbered_markers)

                    # Handle markers class
                    elif class_name == 'markers':
                        # Add to investigation list if not already close to an existing marker
                        should_add = True
                        for existing_marker in markers_to_investigate:
                            distance = get_distance_metres(current_location, existing_marker)
                            if distance < 2.0:  # Don't add if within 2m of existing marker
                                should_add = False
                                break

                        if should_add:
                            markers_to_investigate.append(current_location)
                            logging.info(f"➕ Added marker for circle investigation: {current_location[0]:.7f}, {current_location[1]:.7f}")

                # Check if reached waypoint
                distance = get_distance_metres(current_location, target_location)
                if distance <= 1.5:
                    logging.info(f"✅ Reached waypoint (distance: {distance:.1f}m)")
                    break

                last_check = current_time

            time.sleep(0.02)

        return detections

    except Exception as e:
        logging.error(f"Error during GCP detection collection: {str(e)}")
        return []

def perform_circle_search(vehicle, center_location, detector, video_recorder,
                         numbered_markers, markers_to_investigate, output_dir, radius=1.0, points=8):
    """
    Perform circle search around a marker location.
    """
    try:
        logging.info(f"🔄 Performing {radius}m circle search with {points} points")

        center_lat, center_lon, center_alt = center_location
        circle_detections = []

        for i in range(points):
            angle = (i * 360 / points) * math.pi / 180

            north_offset = radius * math.cos(angle)
            east_offset = radius * math.sin(angle)

            search_location = get_location_metres(center_location, north_offset, east_offset)
            search_lat, search_lon, search_alt = search_location

            logging.info(f"   🎯 Circle point {i+1}/{points}: {search_lat:.7f}, {search_lon:.7f}")

            # Move to circle point
            command_waypoint(vehicle, search_lat, search_lon, center_alt)

            # Wait briefly for movement
            time.sleep(2)

            # Check for detections at this point
            frame = get_camera_frame(video_recorder)
            if frame is not None:
                gcp_detections, _ = detector.detect_gcp_markers_in_frame(frame)

                if gcp_detections:
                    current_location = get_location(vehicle)
                    if current_location:
                        for class_name, center_x, center_y, bbox, confidence in gcp_detections:
                            detection_data = {
                                'class': class_name,
                                'gps_location': current_location,
                                'camera_center': (center_x, center_y),
                                'bbox': bbox,
                                'confidence': confidence,
                                'timestamp': time.time(),
                                'found_in_circle': True
                            }

                            circle_detections.append(detection_data)

                            logging.info(f"🎯 Circle detection: {class_name} at {current_location[0]:.7f}, {current_location[1]:.7f}")

                            # Process detection
                            if class_name == 'marker-numbered':
                                save_numbered_marker(detection_data, frame, output_dir, numbered_markers)
                            elif class_name == 'markers':
                                # Add new marker for investigation if far enough from existing
                                should_add = True
                                for existing_marker in markers_to_investigate:
                                    distance = get_distance_metres(current_location, existing_marker)
                                    if distance < 2.0:
                                        should_add = False
                                        break

                                if should_add:
                                    markers_to_investigate.append(current_location)
                                    logging.info(f"➕ New marker found in circle, added for investigation")

        # Return to center after circle
        command_waypoint(vehicle, center_lat, center_lon, center_alt)
        time.sleep(2)

        return circle_detections

    except Exception as e:
        logging.error(f"Error during circle search: {str(e)}")
        return []

def save_numbered_marker(detection_data, frame, output_dir, numbered_markers):
    """
    Save numbered marker detection to JSON and crop image.
    """
    try:
        marker_id = len(numbered_markers) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Crop the detected region
        x1, y1, x2, y2 = detection_data['bbox']

        # Add some padding to the crop
        padding = 20
        h, w = frame.shape[:2]
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(w, x2 + padding)
        y2_crop = min(h, y2 + padding)

        cropped_image = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        # Save cropped image
        crop_filename = f"numbered_marker_{marker_id:03d}_{timestamp}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, cropped_image)

        # Prepare data for JSON
        marker_data = {
            'id': marker_id,
            'gps_location': {
                'latitude': detection_data['gps_location'][0],
                'longitude': detection_data['gps_location'][1],
                'altitude': detection_data['gps_location'][2]
            },
            'detection': {
                'camera_center': detection_data['camera_center'],
                'bbox': detection_data['bbox'],
                'confidence': detection_data['confidence']
            },
            'timestamp': detection_data['timestamp'],
            'timestamp_str': datetime.fromtimestamp(detection_data['timestamp']).isoformat(),
            'cropped_image_path': crop_filename,
            'found_in_circle': detection_data.get('found_in_circle', False)
        }

        numbered_markers.append(marker_data)

        logging.info(f"💾 Saved numbered marker #{marker_id}: {crop_filename}")

    except Exception as e:
        logging.error(f"Error saving numbered marker: {str(e)}")

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
