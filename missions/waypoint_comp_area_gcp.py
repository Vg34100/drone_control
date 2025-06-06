"""
Competition Area GCP Detection Mission
------------------------------------
Systematic search within a defined 4-corner boundary area for GCP markers.
Implements the hybrid approach: reconnaissance -> boundary definition -> systematic coverage -> adaptive refinement.
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
from detection.camera import initialize_camera, capture_frame, close_camera

# Global variable for mission data (needed for signal handler)
mission_data = None

# Try to import YOLO, fallback if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available - install ultralytics: pip install ultralytics")

class CompetitionAreaGCPDetector:
    """GCP detector optimized for competition area search"""

    def __init__(self, model_path="models/best-gcp.pt", confidence_threshold=0.5,
                 numbered_confidence_threshold=0.4, imgsz=160):
        """Initialize competition GCP detector"""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")

        self.model_path = model_path
        self.general_conf_threshold = confidence_threshold
        self.numbered_conf_threshold = numbered_confidence_threshold
        self.imgsz = imgsz

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GCP model file not found: {model_path}")

        logging.info(f"Loading Competition GCP YOLO model: {self.model_path}")
        logging.info(f"General marker confidence: {self.general_conf_threshold}")
        logging.info(f"Numbered marker confidence: {self.numbered_conf_threshold}")
        self.model = YOLO(self.model_path)

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.detections_count = 0

    def detect_gcp_markers_in_frame(self, frame):
        """
        Detect GCP markers with 2-layer detection system.

        Returns:
            List of detections: [(class_name, center_x, center_y, bbox, confidence), ...]
        """
        if frame is None:
            return [], frame

        try:
            start_time = time.time()

            # Run YOLO inference with general confidence threshold
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.general_conf_threshold,
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

                        # Apply confidence thresholds based on class
                        use_detection = False
                        if class_name == 'marker-numbered' and confidence >= self.numbered_conf_threshold:
                            use_detection = True
                        elif class_name == 'markers' and confidence >= self.general_conf_threshold:
                            use_detection = True

                        if use_detection:
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

            # Perform 2-layer check for 'markers' class
            enhanced_detections = self._enhance_marker_classification(detections, frame)

            # Add frame info
            self._add_frame_info(debug_image, len(enhanced_detections), inference_time)

            return enhanced_detections, debug_image

        except Exception as e:
            logging.error(f"Error in competition GCP detection: {str(e)}")
            return [], frame

    def _enhance_marker_classification(self, detections, frame):
        """
        2-layer detection: check if 'markers' should be 'marker-numbered'
        """
        enhanced_detections = []

        for detection in detections:
            class_name, center_x, center_y, bbox, confidence = detection

            if class_name == 'markers':
                # Crop region and re-run detection with lower threshold for numbered
                x1, y1, x2, y2 = bbox
                padding = 10
                h, w = frame.shape[:2]

                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(w, x2 + padding)
                y2_crop = min(h, y2 + padding)

                cropped_region = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                if cropped_region.size > 0:
                    # Re-run detection on crop with numbered threshold
                    crop_results = self.model.predict(
                        cropped_region,
                        imgsz=self.imgsz,
                        conf=self.numbered_conf_threshold,
                        verbose=False
                    )

                    # Check for numbered markers in crop
                    found_numbered = False
                    best_numbered_conf = 0

                    for result in crop_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                crop_class_id = int(box.cls[0])
                                crop_class_name = result.names[crop_class_id] if hasattr(result, 'names') else f"class_{crop_class_id}"
                                crop_confidence = float(box.conf[0])

                                if crop_class_name == 'marker-numbered' and crop_confidence >= self.numbered_conf_threshold:
                                    found_numbered = True
                                    best_numbered_conf = max(best_numbered_conf, crop_confidence)

                    # Upgrade classification if numbered found
                    if found_numbered:
                        enhanced_detection = ('marker-numbered', center_x, center_y, bbox, max(confidence, best_numbered_conf))
                        logging.debug(f"Enhanced general marker to numbered marker (conf: {best_numbered_conf:.3f})")
                    else:
                        enhanced_detection = detection
                else:
                    enhanced_detection = detection
            else:
                enhanced_detection = detection

            enhanced_detections.append(enhanced_detection)

        return enhanced_detections

    def _draw_detection(self, image, x1, y1, x2, y2, center_x, center_y, class_name, confidence):
        """Draw detection on image with color coding"""
        # Color coding: green for markers, red for marker-numbered
        color = (0, 255, 0) if class_name == 'markers' else (0, 0, 255)
        thickness = 2 if class_name == 'markers' else 3

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw center point
        cv2.circle(image, (center_x, center_y), 8, color, -1)
        cv2.circle(image, (center_x, center_y), 12, (255, 255, 255), 2)

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
        info_text = f"COMP GCP FPS: {fps:.1f} | Detections: {num_detections} | Frame: {self.frame_count}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Add crosshair at frame center
        h, w = image.shape[:2]
        cv2.line(image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
        cv2.line(image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)
        cv2.circle(image, (w//2, h//2), 3, (0, 255, 255), -1)

def mission_competition_area_gcp(vehicle, altitude=8, model_path="models/best-gcp.pt",
                               confidence=0.5, video_recorder=None):
    """
    Execute competition-ready area GCP detection mission within 4-corner boundary.

    Implements hybrid approach:
    1. Initial reconnaissance of boundary area
    2. Boundary definition using the 4 corners
    3. Systematic coverage with lawnmower pattern
    4. Adaptive refinement for detected markers

    Args:
        vehicle: The connected mavlink object
        altitude: Flight altitude in meters
        model_path: Path to GCP YOLO model
        confidence: Detection confidence threshold
        video_recorder: Existing video recorder (optional)

    Returns:
        True if mission completed successfully
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    # Global collections for signal handler access
    global mission_data
    mission_data = {
        'all_gcp_detections': [],
        'numbered_markers': [],
        'output_dir': None,
        'boundary_corners': [],
        'mission_start_time': time.time()
    }

    def signal_handler(signum, frame):
        """Handle Ctrl+C interruption and save data"""
        logging.warning("\n🛑 Mission interrupted by user (Ctrl+C)")
        logging.info("💾 Saving collected data before exit...")

        try:
            if mission_data['output_dir'] and mission_data['all_gcp_detections']:
                # Save whatever data we have collected
                save_all_numbered_markers(mission_data['all_gcp_detections'], mission_data['output_dir'])
                save_mission_summary(mission_data['all_gcp_detections'], mission_data['boundary_corners'], mission_data['output_dir'])
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
        logging.info("=== COMPETITION AREA GCP DETECTION MISSION ===")
        logging.info(f"Flight altitude: {altitude}m")
        logging.info(f"Model: {model_path}")
        logging.info(f"Confidence threshold: {confidence}")

        # Define the 4-corner boundary area (your test area)
        boundary_corners = [
            (35.3482380, -119.1051073),  # Northwest
            (35.3481549, -119.1051114),  # Southwest
            (35.3481462, -119.1046983),  # Southeast
            (35.3482402, -119.1046970),  # Northeast
        ]

        mission_data['boundary_corners'] = boundary_corners

        logging.info("Boundary corners:")
        for i, (lat, lon) in enumerate(boundary_corners):
            logging.info(f"  Corner {i+1}: {lat:.7f}, {lon:.7f}")

        # Initialize competition GCP detector
        detector = CompetitionAreaGCPDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            numbered_confidence_threshold=confidence * 0.8,  # Lower threshold for numbered
            imgsz=160
        )

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/competition_area_gcp_{timestamp}"
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
        logging.info("\n🔍 PHASE 1: INITIAL RECONNAISSANCE")
        reconnaissance_detections = perform_boundary_reconnaissance(
            vehicle, boundary_corners, altitude, detector, video_recorder, output_dir
        )

        if reconnaissance_detections:
            logging.info(f"📊 Reconnaissance found {len(reconnaissance_detections)} potential GCP areas")
            mission_data['all_gcp_detections'].extend(reconnaissance_detections)
        else:
            logging.info("📊 No GCPs detected during reconnaissance")

        # PHASE 2: Systematic Coverage
        logging.info("\n🔍 PHASE 2: SYSTEMATIC COVERAGE")
        coverage_detections = perform_systematic_coverage(
            vehicle, boundary_corners, altitude, detector, video_recorder, output_dir
        )

        if coverage_detections:
            logging.info(f"📊 Systematic coverage found {len(coverage_detections)} additional detections")
            mission_data['all_gcp_detections'].extend(coverage_detections)

        # PHASE 3: Adaptive Refinement
        logging.info("\n🔍 PHASE 3: ADAPTIVE REFINEMENT")
        if mission_data['all_gcp_detections']:
            refinement_detections = perform_adaptive_refinement(
                vehicle, mission_data['all_gcp_detections'], altitude, detector, video_recorder, output_dir
            )

            mission_data['all_gcp_detections'].extend(refinement_detections)

            # Process and save all numbered markers found
            numbered_count = save_all_numbered_markers(mission_data['all_gcp_detections'], output_dir)

            logging.info(f"📊 Final results: {numbered_count} numbered markers saved")
        else:
            logging.info("📊 No GCP detections to refine")

        # Save mission summary
        save_mission_summary(mission_data['all_gcp_detections'], boundary_corners, output_dir)

        # Return to launch
        logging.info("\n🏠 RETURN TO LAUNCH")
        return_to_launch(vehicle)
        wait_for_landing(vehicle)

        logging.info("🎉 COMPETITION AREA GCP MISSION COMPLETED SUCCESSFULLY")
        logging.info(f"📂 Results saved to: {output_dir}")
        return True

    except KeyboardInterrupt:
        # This should be caught by signal handler, but just in case
        logging.warning("Mission interrupted")
        return False
    except Exception as e:
        logging.error(f"Critical error during competition area GCP mission: {str(e)}")
        try:
            # Save whatever data we have before emergency RTL
            if mission_data['output_dir'] and mission_data['all_gcp_detections']:
                save_all_numbered_markers(mission_data['all_gcp_detections'], mission_data['output_dir'])
                save_mission_summary(mission_data['all_gcp_detections'], mission_data['boundary_corners'], mission_data['output_dir'])
                logging.info(f"📂 Partial results saved to: {mission_data['output_dir']}")

            logging.info("Attempting emergency return to launch")
            return_to_launch(vehicle)
        except:
            pass
        return False

def perform_boundary_reconnaissance(vehicle, boundary_corners, altitude, detector, video_recorder, output_dir):
    """
    Phase 1: Quick reconnaissance flight around boundary perimeter
    """
    try:
        logging.info("🚁 Flying boundary reconnaissance pattern")

        detections = []

        # Fly to each corner with detection
        for i, (corner_lat, corner_lon) in enumerate(boundary_corners):
            logging.info(f"📍 Reconnaissance point {i+1}/4: {corner_lat:.7f}, {corner_lon:.7f}")

            # Command waypoint
            command_waypoint(vehicle, corner_lat, corner_lon, altitude)

            # Wait for arrival while detecting
            corner_detections = monitor_flight_with_detection(
                vehicle, (corner_lat, corner_lon), altitude, detector, video_recorder, output_dir, timeout=45
            )

            detections.extend(corner_detections)

            logging.info(f"✅ Corner {i+1} complete, {len(corner_detections)} detections")

            # Brief pause for stability
            time.sleep(2)

        return detections

    except Exception as e:
        logging.error(f"Error during boundary reconnaissance: {str(e)}")
        return []

def perform_systematic_coverage(vehicle, boundary_corners, altitude, detector, video_recorder, output_dir):
    """
    Phase 2: Systematic lawnmower pattern coverage within boundary
    """
    try:
        logging.info("🚁 Executing systematic lawnmower pattern")

        # Calculate boundary box
        lats = [corner[0] for corner in boundary_corners]
        lons = [corner[1] for corner in boundary_corners]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Generate lawnmower pattern with appropriate spacing
        pattern_spacing = 0.000045  # ~5m spacing in degrees (approximate)

        pattern_waypoints = generate_lawnmower_pattern(
            min_lat, max_lat, min_lon, max_lon, pattern_spacing
        )

        logging.info(f"📐 Generated {len(pattern_waypoints)} waypoints for systematic coverage")

        detections = []

        # Execute lawnmower pattern
        for i, (wp_lat, wp_lon) in enumerate(pattern_waypoints):
            logging.info(f"📍 Coverage waypoint {i+1}/{len(pattern_waypoints)}: {wp_lat:.7f}, {wp_lon:.7f}")

            # Command waypoint
            command_waypoint(vehicle, wp_lat, wp_lon, altitude)

            # Monitor flight with detection
            wp_detections = monitor_flight_with_detection(
                vehicle, (wp_lat, wp_lon), altitude, detector, video_recorder, output_dir, timeout=30
            )

            detections.extend(wp_detections)

            if len(wp_detections) > 0:
                logging.info(f"🎯 Waypoint {i+1}: {len(wp_detections)} detections")

        return detections

    except Exception as e:
        logging.error(f"Error during systematic coverage: {str(e)}")
        return []

def perform_adaptive_refinement(vehicle, all_detections, altitude, detector, video_recorder, output_dir):
    """
    Phase 3: Revisit high-confidence detections for detailed analysis
    """
    try:
        logging.info("🔍 Performing adaptive refinement of detections")

        # Group detections by location (remove duplicates within 2m)
        unique_locations = cluster_detections_by_location(all_detections, cluster_distance=2.0)

        logging.info(f"📊 Identified {len(unique_locations)} unique detection clusters")

        refinement_detections = []

        for i, cluster in enumerate(unique_locations):
            # Get the best detection from this cluster
            best_detection = max(cluster, key=lambda x: x.get('confidence', 0))
            cluster_location = best_detection['gps_location']

            logging.info(f"🎯 Refining cluster {i+1}/{len(unique_locations)}: {cluster_location[0]:.7f}, {cluster_location[1]:.7f}")

            # Fly to cluster location
            command_waypoint(vehicle, cluster_location[0], cluster_location[1], altitude)

            if wait_for_waypoint_blocking(vehicle, cluster_location[0], cluster_location[1], altitude, timeout=30):
                # Perform detailed analysis at this location
                detailed_detections = perform_detailed_analysis(
                    vehicle, cluster_location, detector, video_recorder, output_dir
                )

                refinement_detections.extend(detailed_detections)
                logging.info(f"✅ Cluster {i+1} analysis: {len(detailed_detections)} refined detections")
            else:
                logging.warning(f"⚠️ Failed to reach cluster {i+1}")

        return refinement_detections

    except Exception as e:
        logging.error(f"Error during adaptive refinement: {str(e)}")
        return []

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

def monitor_flight_with_detection(vehicle, target_location, altitude, detector, video_recorder, output_dir, timeout=30):
    """
    Monitor flight to waypoint while continuously detecting GCPs and saving numbered markers
    """
    try:
        detections = []
        target_lat, target_lon = target_location
        target_pos = (target_lat, target_lon, 0)

        start_time = time.time()
        detection_interval = 0.2  # Check every 200ms
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
                        # Detect GCPs
                        gcp_detections, _ = detector.detect_gcp_markers_in_frame(frame)

                        for detection_data in gcp_detections:
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

                            logging.info(f"🎯 {class_name}: GPS {current_location[0]:.7f}, {current_location[1]:.7f}, Conf: {confidence:.3f}")

                            # SAVE NUMBERED MARKERS IMMEDIATELY with cropped images
                            if class_name == 'marker-numbered':
                                save_numbered_marker_immediate(detection_dict, frame, output_dir)

                    # Check if reached waypoint
                    distance = get_distance_metres(current_location, target_pos)
                    if distance <= 1.5:
                        logging.debug(f"✅ Reached waypoint (distance: {distance:.1f}m)")
                        break

                last_detection_check = current_time

            time.sleep(0.05)

        return detections

    except Exception as e:
        logging.error(f"Error monitoring flight with detection: {str(e)}")
        return []

def perform_detailed_analysis(vehicle, location, detector, video_recorder, output_dir):
    """
    Perform detailed analysis at a specific location including small circle search
    """
    try:
        logging.info(f"🔬 Detailed analysis at {location[0]:.7f}, {location[1]:.7f}")

        detections = []

        # Hover and analyze current position
        time.sleep(1)
        frame = get_camera_frame(video_recorder)
        if frame is not None:
            gcp_detections, _ = detector.detect_gcp_markers_in_frame(frame)

            for detection_data in gcp_detections:
                class_name, center_x, center_y, bbox, confidence = detection_data

                detection_dict = {
                    'class': class_name,
                    'gps_location': location,
                    'camera_center': (center_x, center_y),
                    'bbox': bbox,
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'detailed_analysis': True
                }

                detections.append(detection_dict)

                # Save numbered markers immediately with cropped images
                if class_name == 'marker-numbered':
                    save_numbered_marker_immediate(detection_dict, frame, output_dir)

        # Small circle search for better coverage
        circle_detections = perform_micro_circle_search(vehicle, location, detector, video_recorder, output_dir, radius=0.5)
        detections.extend(circle_detections)

        return detections

    except Exception as e:
        logging.error(f"Error during detailed analysis: {str(e)}")
        return []

def perform_micro_circle_search(vehicle, center_location, detector, video_recorder, radius=0.5, points=4):
    """
    Small circle search around a detection point
    """
    try:
        logging.info(f"🔄 Micro circle search: {radius}m radius")

        detections = []
        center_lat, center_lon, center_alt = center_location

        for i in range(points):
            angle = (i * 360 / points) * math.pi / 180

            north_offset = radius * math.cos(angle)
            east_offset = radius * math.sin(angle)

            search_location = get_location_metres(center_location, north_offset, east_offset)
            search_lat, search_lon, search_alt = search_location

            # Move to circle point
            command_waypoint(vehicle, search_lat, search_lon, center_alt)
            time.sleep(1.5)  # Brief pause for movement

            # Check for detections
            frame = get_camera_frame(video_recorder)
            if frame is not None:
                gcp_detections, _ = detector.detect_gcp_markers_in_frame(frame)

                if gcp_detections:
                    current_location = get_location(vehicle)
                    for detection_data in gcp_detections:
                        class_name, center_x, center_y, bbox, confidence = detection_data

                        detection_dict = {
                            'class': class_name,
                            'gps_location': current_location or search_location,
                            'camera_center': (center_x, center_y),
                            'bbox': bbox,
                            'confidence': confidence,
                            'timestamp': time.time(),
                            'circle_search': True
                        }

                        detections.append(detection_dict)

        # Return to center
        command_waypoint(vehicle, center_lat, center_lon, center_alt)
        time.sleep(1)

        return detections

    except Exception as e:
        logging.error(f"Error during micro circle search: {str(e)}")
        return []

def cluster_detections_by_location(detections, cluster_distance=2.0):
    """
    Group detections that are close together into clusters
    """
    try:
        if not detections:
            return []

        clusters = []

        for detection in detections:
            if 'gps_location' not in detection:
                continue

            detection_location = detection['gps_location']

            # Find if this detection belongs to an existing cluster
            added_to_cluster = False

            for cluster in clusters:
                # Check distance to any detection in this cluster
                for cluster_detection in cluster:
                    cluster_location = cluster_detection['gps_location']
                    distance = get_distance_metres(detection_location, cluster_location)

                    if distance <= cluster_distance:
                        cluster.append(detection)
                        added_to_cluster = True
                        break

                if added_to_cluster:
                    break

            # If not added to any cluster, create new cluster
            if not added_to_cluster:
                clusters.append([detection])

        return clusters

    except Exception as e:
        logging.error(f"Error clustering detections: {str(e)}")
        return []

def save_numbered_marker_with_image(detection_dict, frame, output_dir):
    """
    Save numbered marker with cropped image
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        marker_id = len([f for f in os.listdir(output_dir) if f.startswith('numbered_marker_')]) + 1

        # Crop image
        x1, y1, x2, y2 = detection_dict['bbox']
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

        logging.info(f"💾 Saved numbered marker #{marker_id}: {crop_filename}")

    except Exception as e:
        logging.error(f"Error saving numbered marker image: {str(e)}")

def save_all_numbered_markers(all_detections, output_dir):
    """
    Save comprehensive JSON summary of all numbered markers found during mission
    """
    try:
        # Filter for numbered markers only
        numbered_detections = [d for d in all_detections if d.get('class') == 'marker-numbered']

        if not numbered_detections:
            logging.info("No numbered markers found to save")
            return 0

        # Create comprehensive results structure
        numbered_markers_summary = []

        for i, detection in enumerate(numbered_detections, 1):
            gps_location = detection.get('gps_location', (0, 0, 0))

            marker_entry = {
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
                },
                'mission_context': {
                    'detection_phase': 'detailed_analysis' if detection.get('detailed_analysis') else
                                    'circle_search' if detection.get('circle_search') else 'systematic_coverage',
                    'detailed_analysis': detection.get('detailed_analysis', False),
                    'circle_search': detection.get('circle_search', False)
                }
            }

            numbered_markers_summary.append(marker_entry)

        # Create final JSON structure
        final_results = {
            'mission_info': {
                'mission_type': 'competition_area_gcp',
                'mission_timestamp': datetime.now().isoformat(),
                'total_numbered_markers_found': len(numbered_detections),
                'mission_duration_seconds': time.time() - (numbered_detections[0].get('timestamp', time.time()) if numbered_detections else time.time())
            },
            'summary_statistics': {
                'total_detections': len(numbered_detections),
                'detections_by_phase': {
                    'systematic_coverage': len([d for d in numbered_detections if not d.get('detailed_analysis') and not d.get('circle_search')]),
                    'detailed_analysis': len([d for d in numbered_detections if d.get('detailed_analysis')]),
                    'circle_search': len([d for d in numbered_detections if d.get('circle_search')])
                },
                'average_confidence': sum(d.get('confidence', 0) for d in numbered_detections) / len(numbered_detections),
                'confidence_range': {
                    'min': min(d.get('confidence', 0) for d in numbered_detections),
                    'max': max(d.get('confidence', 0) for d in numbered_detections)
                }
            },
            'numbered_markers': numbered_markers_summary
        }

        # Save comprehensive results
        json_filename = "numbered_markers_competition_results.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Also save a simple coordinates-only file for quick reference
        simple_coords = [
            {
                'id': i,
                'latitude': marker['gps_location']['latitude'],
                'longitude': marker['gps_location']['longitude'],
                'confidence': marker['detection_details']['confidence']
            }
            for i, marker in enumerate(numbered_markers_summary, 1)
        ]

        simple_filename = "numbered_markers_coordinates_only.json"
        simple_path = os.path.join(output_dir, simple_filename)

        with open(simple_path, 'w') as f:
            json.dump({
                'total_markers': len(simple_coords),
                'coordinates': simple_coords
            }, f, indent=2)

        logging.info(f"💾 Saved {len(numbered_detections)} numbered markers to:")
        logging.info(f"  📄 Comprehensive: {json_filename}")
        logging.info(f"  📄 Simple: {simple_filename}")

        return len(numbered_detections)

    except Exception as e:
        logging.error(f"Error saving numbered markers summary: {str(e)}")
        return 0

def save_mission_summary(all_detections, boundary_corners, output_dir):
    """
    Save comprehensive mission summary
    """
    try:
        # Count detections by type and phase
        numbered_count = sum(1 for d in all_detections if d.get('class') == 'marker-numbered')
        markers_count = sum(1 for d in all_detections if d.get('class') == 'markers')

        reconnaissance_count = sum(1 for d in all_detections if not d.get('detailed_analysis') and not d.get('circle_search'))
        detailed_count = sum(1 for d in all_detections if d.get('detailed_analysis'))
        circle_count = sum(1 for d in all_detections if d.get('circle_search'))

        summary = {
            'mission_info': {
                'mission_type': 'competition_area_gcp',
                'timestamp': datetime.now().isoformat(),
                'boundary_area': {
                    'corners': [{'lat': corner[0], 'lon': corner[1]} for corner in boundary_corners]
                }
            },
            'detection_summary': {
                'total_detections': len(all_detections),
                'numbered_markers': numbered_count,
                'general_markers': markers_count,
                'by_phase': {
                    'reconnaissance': reconnaissance_count,
                    'systematic_coverage': len(all_detections) - detailed_count - circle_count - reconnaissance_count,
                    'detailed_analysis': detailed_count,
                    'circle_search': circle_count
                }
            },
            'all_detections': [
                {
                    'class': d.get('class', 'unknown'),
                    'gps_location': d.get('gps_location', (0, 0, 0)),
                    'confidence': d.get('confidence', 0.0),
                    'timestamp': d.get('timestamp', 0),
                    'detection_phase': 'reconnaissance' if not d.get('detailed_analysis') and not d.get('circle_search') else
                                    'detailed_analysis' if d.get('detailed_analysis') else 'circle_search'
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

def perform_micro_circle_search(vehicle, center_location, detector, video_recorder, output_dir, radius=0.5, points=4):
    """
    Small circle search around a detection point
    """
    try:
        logging.info(f"🔄 Micro circle search: {radius}m radius")

        detections = []
        center_lat, center_lon, center_alt = center_location

        for i in range(points):
            angle = (i * 360 / points) * math.pi / 180

            north_offset = radius * math.cos(angle)
            east_offset = radius * math.sin(angle)

            search_location = get_location_metres(center_location, north_offset, east_offset)
            search_lat, search_lon, search_alt = search_location

            # Move to circle point
            command_waypoint(vehicle, search_lat, search_lon, center_alt)
            time.sleep(1.5)  # Brief pause for movement

            # Check for detections
            frame = get_camera_frame(video_recorder)
            if frame is not None:
                gcp_detections, _ = detector.detect_gcp_markers_in_frame(frame)

                if gcp_detections:
                    current_location = get_location(vehicle)
                    for detection_data in gcp_detections:
                        class_name, center_x, center_y, bbox, confidence = detection_data

                        detection_dict = {
                            'class': class_name,
                            'gps_location': current_location or search_location,
                            'camera_center': (center_x, center_y),
                            'bbox': bbox,
                            'confidence': confidence,
                            'timestamp': time.time(),
                            'circle_search': True
                        }

                        detections.append(detection_dict)

                        # Save numbered markers immediately with cropped images
                        if class_name == 'marker-numbered':
                            save_numbered_marker_immediate(detection_dict, frame, output_dir)

        # Return to center
        command_waypoint(vehicle, center_lat, center_lon, center_alt)
        time.sleep(1)

        return detections

    except Exception as e:
        logging.error(f"Error during micro circle search: {str(e)}")
        return []

def save_numbered_marker_immediate(detection_dict, frame, output_dir):
    """
    Immediately save numbered marker with cropped image and metadata
    """
    try:
        # Generate unique timestamp and ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Count existing numbered marker files to get next ID
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('numbered_marker_') and f.endswith('.jpg')]
        marker_id = len(existing_files) + 1

        # Validate detection data
        if 'bbox' not in detection_dict or 'gps_location' not in detection_dict:
            logging.error("Invalid detection data for saving numbered marker")
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
        crop_filename = f"numbered_marker_{marker_id:03d}_{timestamp}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save image with high quality
        success = cv2.imwrite(crop_path, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not success:
            logging.error(f"Failed to save cropped image: {crop_filename}")
            return False

        # Create metadata file alongside image
        metadata_filename = f"numbered_marker_{marker_id:03d}_{timestamp}.json"
        metadata_path = os.path.join(output_dir, metadata_filename)

        gps_location = detection_dict['gps_location']
        marker_metadata = {
            'marker_id': marker_id,
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
            },
            'detection_context': {
                'detailed_analysis': detection_dict.get('detailed_analysis', False),
                'circle_search': detection_dict.get('circle_search', False),
                'phase': 'detailed_analysis' if detection_dict.get('detailed_analysis') else
                        'circle_search' if detection_dict.get('circle_search') else 'reconnaissance'
            }
        }

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(marker_metadata, f, indent=2)

        logging.info(f"💾 Saved numbered marker #{marker_id}: {crop_filename}")
        logging.info(f"📍 GPS: {gps_location[0]:.7f}, {gps_location[1]:.7f} | Conf: {detection_dict.get('confidence', 0.0):.3f}")

        return True

    except Exception as e:
        logging.error(f"Error saving numbered marker: {str(e)}")
        return False

def save_numbered_marker_with_image(detection_dict, frame, output_dir):
    """
    Legacy function name - redirects to immediate save function
    """
    return save_numbered_marker_immediate(detection_dict, frame, output_dir)# missions/waypoint_comp_area_gcp.py - NEW FILE
