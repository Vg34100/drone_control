# detection/gcp_detector.py - NEW MODULE
"""
GCP (Ground Control Point) Detection Test Module
-----------------------------------------------
Comprehensive testing module for GCP marker detection including X-patterns,
squares, triangles, and other geometric markers. Designed for drone competition use.
"""

import cv2
import numpy as np
import logging
import time
import os
import math
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path

class GCPDetector:
    """GCP detector for testing various ground control point markers"""

    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the GCP detector.

        Args:
            confidence_threshold: Minimum confidence for X-pattern detections
        """
        self.conf_threshold = confidence_threshold

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.detections_count = 0
        self.detection_stats = {
            'x_patterns': 0,
            'squares': 0,
            'triangles': 0,
            'tri_in_squares': 0
        }

        # X-pattern saving
        self.x_pattern_save_count = 0

    def detect_squares(self, frame, min_area=100, max_area=10000, aspect_ratio_range=(0.8, 1.2)):
        """Detect squares in the frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            squares = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                        squares.append((x, y, w, h, approx))

            return squares
        except Exception as e:
            logging.error(f"Error detecting squares: {str(e)}")
            return []

    def detect_triangles(self, frame, min_area=100, max_area=10000, tolerance=0.15):
        """Detect triangles in the frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            triangles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

                if len(approx) == 3:
                    side1 = np.linalg.norm(approx[0][0] - approx[1][0])
                    side2 = np.linalg.norm(approx[1][0] - approx[2][0])
                    side3 = np.linalg.norm(approx[2][0] - approx[0][0])

                    avg_side = (side1 + side2 + side3) / 3

                    if (abs(side1 - avg_side) / avg_side < tolerance and
                        abs(side2 - avg_side) / avg_side < tolerance and
                        abs(side3 - avg_side) / avg_side < tolerance):

                        x, y, w, h = cv2.boundingRect(approx)
                        triangles.append((x, y, w, h, approx))

            return triangles
        except Exception as e:
            logging.error(f"Error detecting triangles: {str(e)}")
            return []

    def detect_x_pattern(self, frame, min_area=500, max_area=100000, diagonal_angle_tolerance=15):
        """Detect X-patterns in the frame (squares with diagonal lines)."""
        try:
            squares = self.detect_squares(frame, min_area, max_area)
            x_patterns = []

            for x, y, w, h, approx in squares:
                # Create a mask for this square
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [approx], 0, 255, -1)

                # Get the ROI
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = cv2.bitwise_and(gray, gray, mask=mask)

                if roi[mask > 0].size == 0:
                    continue

                # Apply edge detection to the ROI
                roi_edges = cv2.Canny(roi, 30, 120)

                # Look for diagonal lines using Hough transform
                lines = cv2.HoughLinesP(
                    roi_edges, 1, np.pi/180,
                    threshold=15,
                    minLineLength=20,
                    maxLineGap=20
                )

                diagonal_lines = []
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

                        if ((45 - diagonal_angle_tolerance <= angle <= 45 + diagonal_angle_tolerance) or
                            (135 - diagonal_angle_tolerance <= angle <= 135 + diagonal_angle_tolerance)):
                            diagonal_lines.append((x1, y1, x2, y2, angle))

                # Check quadrant intensity differences
                h_half, w_half = h // 2, w // 2
                quadrant_pattern_detected = False

                if h_half > 0 and w_half > 0:
                    q1 = roi[y:y+h_half, x:x+w_half]
                    q2 = roi[y:y+h_half, x+w_half:x+w]
                    q3 = roi[y+h_half:y+h, x:x+w_half]
                    q4 = roi[y+h_half:y+h, x+w_half:x+w]

                    q1_intensity = np.mean(q1[q1 > 0]) if np.any(q1 > 0) else 0
                    q2_intensity = np.mean(q2[q2 > 0]) if np.any(q2 > 0) else 0
                    q3_intensity = np.mean(q3[q3 > 0]) if np.any(q3 > 0) else 0
                    q4_intensity = np.mean(q4[q4 > 0]) if np.any(q4 > 0) else 0

                    diagonal1_diff = abs(q1_intensity - q4_intensity)
                    diagonal2_diff = abs(q2_intensity - q3_intensity)
                    cross_diff = abs((q1_intensity + q4_intensity) / 2 - (q2_intensity + q3_intensity) / 2)

                    quadrant_pattern_detected = (
                        diagonal1_diff < 10 and
                        diagonal2_diff < 10 and
                        cross_diff > 20
                    )

                has_enough_diagonals = len(diagonal_lines) >= 2

                if has_enough_diagonals or quadrant_pattern_detected:
                    confidence = (len(diagonal_lines) * 0.3) + (1 if quadrant_pattern_detected else 0)
                    confidence = min(confidence, 1.0)  # Cap at 1.0

                    if confidence >= self.conf_threshold:
                        x_patterns.append((x, y, w, h, confidence, diagonal_lines))

                        # Save preprocessing steps for X-pattern debugging
                        if self.x_pattern_save_count % 3 == 0:  # Save every 3rd X-pattern detection
                            self._save_x_pattern_debug(frame, x, y, w, h, confidence, gray, roi, roi_edges, diagonal_lines)
                        self.x_pattern_save_count += 1

            return x_patterns
        except Exception as e:
            logging.error(f"Error detecting X-patterns: {str(e)}")
            return []

    def detect_triangles_in_squares(self, frame, min_triangles=4):
        """Detect squares containing multiple triangles."""
        try:
            triangles = self.detect_triangles(frame)
            squares = self.detect_squares(frame)

            patterns = []
            for sx, sy, sw, sh, s_approx in squares:
                triangles_inside = 0

                for tx, ty, tw, th, t_approx in triangles:
                    t_center_x = tx + tw/2
                    t_center_y = ty + th/2

                    if (sx <= t_center_x <= sx + sw and
                        sy <= t_center_y <= sy + sh):
                        triangles_inside += 1

                if triangles_inside >= min_triangles:
                    patterns.append((sx, sy, sw, sh, triangles_inside))

            return patterns
        except Exception as e:
            logging.error(f"Error detecting triangles in squares: {str(e)}")
            return []

    def detect_gcp_markers_in_frame(self, frame):
        """
        Detect all types of GCP markers in a frame.

        Args:
            frame: Input BGR image

        Returns:
            Tuple of (gcp_markers, debug_image)
            gcp_markers: Dictionary with all detected marker types
            debug_image: Annotated image showing detections
        """
        if frame is None:
            return {}, frame

        try:
            start_time = time.time()

            # Detect different types of markers
            squares = self.detect_squares(frame)
            triangles = self.detect_triangles(frame)
            x_patterns = self.detect_x_pattern(frame)
            tri_in_squares = self.detect_triangles_in_squares(frame)

            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1

            # Update detection counts
            self.detection_stats['squares'] += len(squares)
            self.detection_stats['triangles'] += len(triangles)
            self.detection_stats['x_patterns'] += len(x_patterns)
            self.detection_stats['tri_in_squares'] += len(tri_in_squares)
            self.detections_count += len(squares) + len(triangles) + len(x_patterns) + len(tri_in_squares)

            # Create debug image
            debug_image = frame.copy()

            # Draw detections on the display frame
            for x, y, w, h, _ in squares:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_image, "Square", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for x, y, w, h, _ in triangles:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(debug_image, "Triangle", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for x, y, w, h, conf, lines in x_patterns:
                # Draw main bounding box in red for X-patterns
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

                # Draw detected diagonal lines
                for line_data in lines:
                    x1, y1, x2, y2, angle = line_data
                    cv2.line(debug_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Draw center point
                center_x, center_y = x + w//2, y + h//2
                cv2.circle(debug_image, (center_x, center_y), 8, (0, 0, 255), -1)
                cv2.circle(debug_image, (center_x, center_y), 12, (255, 255, 255), 2)

                # Add confidence label
                label = f"X-Pattern: {conf:.3f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(debug_image, (x, y - label_size[1] - 15),
                             (x + label_size[0], y - 5), (0, 0, 255), -1)
                cv2.putText(debug_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for x, y, w, h, count in tri_in_squares:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(debug_image, f"{count} triangles", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Add frame info to debug image
            self._add_frame_info(debug_image, inference_time)

            # Prepare return data
            gcp_markers = {
                'squares': squares,
                'triangles': triangles,
                'x_patterns': x_patterns,
                'tri_in_squares': tri_in_squares,
                'total_detections': len(squares) + len(triangles) + len(x_patterns) + len(tri_in_squares)
            }

            return gcp_markers, debug_image

        except Exception as e:
            logging.error(f"Error in GCP marker detection: {str(e)}")
            return {}, frame

    def _add_frame_info(self, image, inference_time):
        """Add frame information overlay"""
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Add frame info
        info_text = f"FPS: {fps:.1f} | Total Detections: {self.detections_count} | Frame: {self.frame_count}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Add detection breakdown
        breakdown = f"X:{self.detection_stats['x_patterns']} Sq:{self.detection_stats['squares']} Tri:{self.detection_stats['triangles']} TriSq:{self.detection_stats['tri_in_squares']}"
        cv2.putText(image, breakdown, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, breakdown, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add crosshair at frame center for drone alignment reference
        h, w = image.shape[:2]
        cv2.line(image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
        cv2.line(image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)
        cv2.circle(image, (w//2, h//2), 3, (0, 255, 255), -1)

    def _save_x_pattern_debug(self, original_frame, x, y, w, h, confidence, gray, roi, roi_edges, diagonal_lines):
        """Save debugging images for X-pattern detection"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            debug_dir = "debug_frames"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            # 1. Save original detection
            detection_frame = original_frame.copy()
            cv2.rectangle(detection_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(detection_frame, f"X-Pattern: {confidence:.3f}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(f"{debug_dir}/x_pattern_{self.x_pattern_save_count}_{timestamp}_detection.jpg", detection_frame)

            # 2. Save ROI preprocessing steps
            cv2.imwrite(f"{debug_dir}/x_pattern_{self.x_pattern_save_count}_{timestamp}_roi.jpg", roi)
            cv2.imwrite(f"{debug_dir}/x_pattern_{self.x_pattern_save_count}_{timestamp}_edges.jpg", roi_edges)

            # 3. Save detected lines overlay
            lines_frame = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            for line_data in diagonal_lines:
                x1, y1, x2, y2, angle = line_data
                cv2.line(lines_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(lines_frame, f"{angle:.0f}°", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.imwrite(f"{debug_dir}/x_pattern_{self.x_pattern_save_count}_{timestamp}_lines.jpg", lines_frame)

            logging.info(f"Saved X-pattern debug images: x_pattern_{self.x_pattern_save_count}_{timestamp}_*.jpg")

        except Exception as e:
            logging.warning(f"Failed to save X-pattern debug images: {str(e)}")

    def get_performance_stats(self):
        """Get performance statistics"""
        if self.frame_count == 0:
            return {}

        avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
        avg_inference_ms = (self.total_inference_time / self.frame_count) * 1000

        return {
            'total_frames': self.frame_count,
            'total_detections': self.detections_count,
            'avg_detections_per_frame': self.detections_count / self.frame_count,
            'avg_fps': avg_fps,
            'avg_inference_ms': avg_inference_ms,
            'detection_breakdown': self.detection_stats.copy()
        }

def create_enhanced_gcp_display(original_frame, gcp_markers, debug_image):
    """
    Create enhanced display with big indicators for GCP detections.
    """
    enhanced = debug_image.copy()
    height, width = enhanced.shape[:2]

    total_detections = gcp_markers.get('total_detections', 0)
    x_patterns = gcp_markers.get('x_patterns', [])

    # Add detection indicators
    if total_detections > 0:
        # Determine primary detection type for color coding
        if len(x_patterns) > 0:
            # X-patterns found - highest priority (red theme)
            overlay = enhanced.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 150), -1)
            enhanced = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)

            cv2.putText(enhanced, f"🎯 X-PATTERN DETECTED! ({len(x_patterns)} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(enhanced, f"🎯 X-PATTERN DETECTED! ({len(x_patterns)} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Add confidence info for X-patterns
            best_x_pattern = max(x_patterns, key=lambda x: x[4])  # Get highest confidence
            conf_text = f"Best Confidence: {best_x_pattern[4]:.1%}"
            cv2.putText(enhanced, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:
            # Other GCP markers found (green theme)
            overlay = enhanced.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 150, 0), -1)
            enhanced = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)

            cv2.putText(enhanced, f"📍 GCP MARKERS DETECTED! ({total_detections} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(enhanced, f"📍 GCP MARKERS DETECTED! ({total_detections} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Draw big circles around X-patterns (highest priority)
        for x, y, w, h, conf, lines in x_patterns:
            center_x, center_y = x + w//2, y + h//2
            # Big outer circle for X-patterns
            cv2.circle(enhanced, (center_x, center_y), 100, (0, 0, 255), 5)
            # Inner circle
            cv2.circle(enhanced, (center_x, center_y), 50, (255, 255, 255), 3)

    else:
        # No detections found
        overlay = enhanced.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 150), -1)
        enhanced = cv2.addWeighted(enhanced, 0.8, overlay, 0.2, 0)

        cv2.putText(enhanced, "🔍 NO GCP MARKERS DETECTED",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(enhanced, "🔍 NO GCP MARKERS DETECTED",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return enhanced

def test_gcp_detection(source: Union[str, int] = 0,
                      display: bool = True,
                      save_results: bool = True,
                      duration: float = 0,
                      video_delay: float = 0.1,
                      confidence: float = 0.5) -> bool:
    """
    Test GCP detection on camera feed, video file, or image.

    Args:
        source: Camera ID (int), video file path, or image file path
        display: Whether to display the detection results
        save_results: Whether to save detection results
        duration: Duration for camera/video (0 = until 'q' pressed)
        video_delay: Delay between frames for video/camera (seconds)
        confidence: Confidence threshold for X-pattern detections

    Returns:
        True if test completed successfully
    """
    try:
        logging.info(f"Starting GCP detection test with source: {source}")

        detector = GCPDetector(confidence_threshold=confidence)

        # Determine source type
        if isinstance(source, int):
            # Camera input
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logging.error(f"Failed to open camera {source}")
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            source_type = "camera"

        elif isinstance(source, str):
            if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Image file
                image = cv2.imread(source)
                if image is None:
                    logging.error(f"Failed to load image: {source}")
                    return False
                source_type = "image"
            else:
                # Video file
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    logging.error(f"Failed to open video: {source}")
                    return False
                source_type = "video"
        else:
            logging.error("Invalid source type")
            return False

        # Process single image
        if source_type == "image":
            gcp_markers, debug_image = detector.detect_gcp_markers_in_frame(image)

            # Create enhanced display image
            enhanced_image = create_enhanced_gcp_display(image, gcp_markers, debug_image)

            logging.info(f"Detected {gcp_markers['total_detections']} GCP markers in image")
            for marker_type, markers in gcp_markers.items():
                if marker_type != 'total_detections' and markers:
                    logging.info(f"  {marker_type}: {len(markers)}")

            if display:
                cv2.imshow("GCP Detection Results", enhanced_image)
                print("\n" + "="*60)
                print("📸 GCP IMAGE DETECTION RESULTS")
                print("="*60)
                if gcp_markers['total_detections'] > 0:
                    print(f"🎯 GCP MARKERS FOUND: {gcp_markers['total_detections']}")
                    for marker_type, markers in gcp_markers.items():
                        if marker_type != 'total_detections' and markers:
                            print(f"   {marker_type}: {len(markers)}")
                            if marker_type == 'x_patterns':
                                for i, (x, y, w, h, conf, _) in enumerate(markers):
                                    print(f"     X-Pattern {i+1}: Position ({x+w//2}, {y+h//2}), Confidence: {conf:.1%}")
                else:
                    print("❌ NO GCP MARKERS DETECTED")
                print("="*60)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_results:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gcp_detection_{timestamp}.jpg"
                cv2.imwrite(filename, enhanced_image)
                logging.info(f"Results saved to: {filename}")

            return True

        # Process video or camera feed
        else:
            total_frames = 0
            detection_frames = 0
            start_time = time.time()

            if source_type == "video":
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"Video: {total_video_frames} frames @ {video_fps} FPS")

            logging.info(f"Processing {source_type}. Press 'q' to quit, 's' to save frame")
            print("\n" + "="*80)
            print(f"🎥 {source_type.upper()} GCP DETECTION")
            print("="*80)
            print("Controls: 'q' = quit, 's' = save frame, 'p' = pause")
            print("="*80)

            while True:
                ret, frame = cap.read()

                if not ret:
                    if source_type == "camera":
                        logging.error("Failed to capture frame")
                        break
                    else:
                        logging.info("End of video reached")
                        break

                total_frames += 1

                # Detect GCP markers
                gcp_markers, debug_image = detector.detect_gcp_markers_in_frame(frame)

                # Create enhanced display
                enhanced_image = create_enhanced_gcp_display(frame, gcp_markers, debug_image)

                # Track detections
                if gcp_markers['total_detections'] > 0:
                    detection_frames += 1
                    print(f"🎯 Frame {total_frames}: {gcp_markers['total_detections']} GCP marker(s) detected!")

                    # Show breakdown
                    for marker_type, markers in gcp_markers.items():
                        if marker_type != 'total_detections' and markers:
                            print(f"   → {marker_type}: {len(markers)}")
                            if marker_type == 'x_patterns':
                                for i, (x, y, w, h, conf, _) in enumerate(markers):
                                    print(f"     X-Pattern {i+1} at ({x+w//2}, {y+h//2}), confidence: {conf:.3f}")

                elif total_frames % 30 == 0:  # Log every 30 frames when no detection
                    print(f"🔍 Frame {total_frames}: Searching for GCP markers...")

                # Display results
                if display:
                    cv2.imshow("GCP Detection", enhanced_image)

                    key = cv2.waitKey(int(video_delay * 1000)) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_results:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"gcp_frame_{total_frames}_{timestamp}.jpg"
                        cv2.imwrite(filename, enhanced_image)
                        logging.info(f"Frame saved: {filename}")
                    elif key == ord('p'):  # Pause
                        cv2.waitKey(0)
                else:
                    time.sleep(video_delay)

                # Check duration for camera
                if source_type == "camera" and duration > 0:
                    if time.time() - start_time >= duration:
                        break

                # Progress for video
                if source_type == "video" and total_frames % 30 == 0:
                    progress = (total_frames / total_video_frames) * 100
                    print(f"Progress: {progress:.1f}% ({total_frames}/{total_video_frames})")

            # Cleanup
            cap.release()
            if display:
                cv2.destroyAllWindows()

            # Final summary
            total_time = time.time() - start_time
            detection_rate = (detection_frames / total_frames * 100) if total_frames > 0 else 0
            stats = detector.get_performance_stats()

            print("\n" + "="*80)
            print("📊 GCP DETECTION SUMMARY")
            print("="*80)
            print(f"Total frames processed: {total_frames}")
            print(f"Frames with GCP markers: {detection_frames}")
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"Processing time: {total_time:.1f}s")
            print(f"Average FPS: {stats.get('avg_fps', 0):.1f}")
            print(f"Average inference time: {stats.get('avg_inference_ms', 0):.1f}ms")

            # Detection breakdown
            breakdown = stats.get('detection_breakdown', {})
            print(f"\nDetection Breakdown:")
            for marker_type, count in breakdown.items():
                if count > 0:
                    print(f"  {marker_type}: {count}")

            if detection_frames > 0:
                print(f"🎯 SUCCESS: GCP markers detected in {detection_frames} frames!")
                if breakdown.get('x_patterns', 0) > 0:
                    print(f"🎯 X-PATTERNS FOUND: {breakdown['x_patterns']} total detections!")
            else:
                print("❌ NO GCP MARKERS DETECTED in any frame")
            print("="*80)

            return True

    except Exception as e:
        logging.error(f"Error during GCP detection test: {str(e)}")
        return False

def create_gcp_detector(confidence=0.5):
    """
    Factory function to create a GCP detector.

    Returns:
        GCPDetector instance
    """
    return GCPDetector(confidence)

# For backwards compatibility with existing gcp.py functions
def detect_gcp_markers(frame, save_debug=False, debug_dir="debug_frames"):
    """
    Backwards compatible function for existing code.
    """
    detector = GCPDetector()
    gcp_markers, debug_image = detector.detect_gcp_markers_in_frame(frame)

    # Convert to old format for compatibility
    result = {
        'display_frame': debug_image,
        'squares': gcp_markers.get('squares', []),
        'triangles': gcp_markers.get('triangles', []),
        'x_patterns': gcp_markers.get('x_patterns', []),
        'tri_in_squares': gcp_markers.get('tri_in_squares', [])
    }

    if save_debug and debug_dir:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{debug_dir}/gcp_detection_{timestamp}.jpg", debug_image)

    return result
