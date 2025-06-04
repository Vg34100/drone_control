# detection/gcp_yolo_detector.py - UPDATED WITH LAYERED DETECTION
"""
GCP YOLO Detection Test Module - Enhanced with Layered Detection
---------------------------------------------------------------
Updated to perform layered detection: first check for numbered markers,
then fallback to general markers if confidence is low.
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available - install ultralytics: pip install ultralytics")

class GCPYOLODetector:
    """YOLO-based GCP detector with layered detection system"""

    def __init__(self, model_path="models/best-gcp.pt", confidence_threshold=0.5,
                 numbered_confidence_threshold=0.4, imgsz=160):
        """
        Initialize the GCP detector with YOLO model.

        Args:
            model_path: Path to GCP YOLO model file
            confidence_threshold: Minimum confidence for general marker detections
            numbered_confidence_threshold: Lower threshold for numbered marker detection layer
            imgsz: Model input image size
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")

        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.numbered_conf_threshold = numbered_confidence_threshold
        self.imgsz = imgsz

        # Try to load the model
        if not os.path.exists(model_path):
            logging.error(f"GCP model file not found: {model_path}")
            # Try common locations
            alt_paths = [
                "best-gcp.pt",
                "models/best-gcp.pt",
                "best-gcp.engine",
                "models/best-gcp.engine"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.model_path = alt_path
                    logging.info(f"Found GCP model at: {alt_path}")
                    break
            else:
                raise FileNotFoundError(f"Could not find GCP model file. Tried: {model_path}, {alt_paths}")

        logging.info(f"Loading GCP YOLO model: {self.model_path}")
        logging.info(f"General marker confidence threshold: {self.conf_threshold}")
        logging.info(f"Numbered marker confidence threshold: {self.numbered_conf_threshold}")
        self.model = YOLO(self.model_path)

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.detections_count = 0
        self.layered_detection_count = 0

    def detect_gcp_markers_in_frame(self, frame):
        """
        Detect GCP markers in a single frame using improved detection with NMS.

        IMPROVED LOGIC:
        1. Run inference with standard confidence threshold
        2. Apply Non-Maximum Suppression to remove overlapping detections
        3. For 'markers' class detections, also check if they might be numbered markers
        4. Keep only the best detection per physical marker

        Args:
            frame: Input BGR image

        Returns:
            Tuple of (gcp_markers, debug_image)
            gcp_markers: List of (class_name, center_x, center_y, bbox_info, confidence)
            debug_image: Annotated image showing detections
        """
        if frame is None:
            return [], frame

        try:
            start_time = time.time()

            # Run inference with standard confidence threshold
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False
            )

            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1

            # Process results and remove overlapping detections
            all_detections = []
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

                        # Only keep detections that meet confidence requirements
                        if class_name == 'marker-numbered' and confidence >= self.numbered_conf_threshold:
                            use_detection = True
                        elif class_name == 'markers' and confidence >= self.conf_threshold:
                            use_detection = True
                        else:
                            use_detection = False

                        if use_detection:
                            # Calculate center
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            # Store detection for overlap removal
                            all_detections.append({
                                'class': class_name,
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'confidence': confidence,
                                'area': (x2 - x1) * (y2 - y1)
                            })

            # Remove overlapping detections using distance-based NMS
            final_detections = self._remove_overlapping_detections(all_detections)

            # LAYERED CHECK: For remaining 'markers', see if they should be 'marker-numbered'
            enhanced_detections = self._enhance_marker_classification(final_detections, frame)

            # Convert to final format and draw
            gcp_markers = []
            for detection in enhanced_detections:
                class_name = detection['class']
                center_x, center_y = detection['center']
                bbox = detection['bbox']
                confidence = detection['confidence']

                bbox_info = {
                    'bbox': bbox,
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1],
                    'area': detection['area']
                }

                gcp_markers.append((class_name, center_x, center_y, bbox_info, confidence))
                self.detections_count += 1

                # Draw on debug image
                x1, y1, x2, y2 = bbox
                self._draw_detection(debug_image, x1, y1, x2, y2, center_x, center_y,
                                   class_name, confidence)

            # Add frame info to debug image
            self._add_frame_info(debug_image, len(gcp_markers), inference_time)

            return gcp_markers, debug_image

        except Exception as e:
            logging.error(f"Error in GCP YOLO detection: {str(e)}")
            return [], frame

    def _remove_overlapping_detections(self, detections, overlap_threshold=0.3):
        """
        Remove overlapping detections using distance-based approach.

        Args:
            detections: List of detection dictionaries
            overlap_threshold: IoU threshold for considering detections as overlapping

        Returns:
            List of non-overlapping detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        final_detections = []

        for current_det in sorted_detections:
            is_overlapping = False

            for kept_det in final_detections:
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(current_det['bbox'], kept_det['bbox'])

                if iou > overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                final_detections.append(current_det)

        logging.debug(f"Removed {len(detections) - len(final_detections)} overlapping detections")
        return final_detections

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _enhance_marker_classification(self, detections, frame):
        """
        For 'markers' class detections, check if they should be 'marker-numbered'.
        This runs a secondary check with lower confidence on the cropped region.
        """
        enhanced_detections = []

        for detection in detections:
            if detection['class'] == 'markers':
                # Crop the region and re-run detection with lower threshold
                x1, y1, x2, y2 = detection['bbox']

                # Add padding to crop
                padding = 10
                h, w = frame.shape[:2]
                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(w, x2 + padding)
                y2_crop = min(h, y2 + padding)

                cropped_region = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                if cropped_region.size > 0:
                    # Re-run detection on cropped region with lower threshold
                    crop_results = self.model.predict(
                        cropped_region,
                        imgsz=self.imgsz,
                        conf=self.numbered_conf_threshold,
                        verbose=False
                    )

                    # Check if any numbered markers found in crop
                    found_numbered = False
                    best_numbered_conf = 0

                    for result in crop_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                                confidence = float(box.conf[0])

                                if class_name == 'marker-numbered' and confidence >= self.numbered_conf_threshold:
                                    found_numbered = True
                                    best_numbered_conf = max(best_numbered_conf, confidence)

                    # If numbered marker found in crop, upgrade the classification
                    if found_numbered:
                        detection['class'] = 'marker-numbered'
                        detection['confidence'] = max(detection['confidence'], best_numbered_conf)
                        self.layered_detection_count += 1
                        logging.debug(f"Enhanced general marker to numbered marker (conf: {best_numbered_conf:.3f})")

            enhanced_detections.append(detection)

        return enhanced_detections

    def _draw_detection(self, image, x1, y1, x2, y2, center_x, center_y, class_name, confidence):
        """Draw detection on image"""
        # Color coding: green for 'markers', red for 'marker-numbered'
        if class_name == 'markers':
            color = (0, 255, 0)  # Green
            thickness = 2
        elif class_name == 'marker-numbered':
            color = (0, 0, 255)  # Red
            thickness = 3
        else:
            color = (255, 0, 255)  # Magenta for unknown classes
            thickness = 2

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw center point
        cv2.circle(image, (center_x, center_y), 8, color, -1)
        cv2.circle(image, (center_x, center_y), 12, (255, 255, 255), 2)

        # Draw crosshair at center
        cv2.line(image, (center_x - 15, center_y), (center_x + 15, center_y), color, 2)
        cv2.line(image, (center_x, center_y - 15), (center_x, center_y + 15), color, 2)

        # Add confidence label
        label = f"{class_name}: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _add_frame_info(self, image, num_detections, inference_time):
        """Add frame information overlay"""
        # Calculate FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Add frame info
        info_text = f"GCP FPS: {fps:.1f} | Detections: {num_detections} | Frame: {self.frame_count}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Add detection breakdown with correct counts
        # Note: gcp_markers should be passed to this function to get accurate counts
        breakdown = f"Total: {num_detections} | Layered Enhancements: {self.layered_detection_count}"
        cv2.putText(image, breakdown, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, breakdown, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add crosshair at frame center for drone alignment reference
        h, w = image.shape[:2]
        cv2.line(image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
        cv2.line(image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)
        cv2.circle(image, (w//2, h//2), 3, (0, 255, 255), -1)

    def get_performance_stats(self):
        """Get performance statistics"""
        if self.frame_count == 0:
            return {}

        avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
        avg_inference_ms = (self.total_inference_time / self.frame_count) * 1000

        return {
            'total_frames': self.frame_count,
            'total_detections': self.detections_count,
            'layered_detections': self.layered_detection_count,
            'avg_detections_per_frame': self.detections_count / self.frame_count,
            'avg_fps': avg_fps,
            'avg_inference_ms': avg_inference_ms
        }

def create_enhanced_gcp_display(original_frame, gcp_markers, debug_image):
    """
    Create enhanced display with big indicators for GCP detections.
    """
    enhanced = debug_image.copy()
    height, width = enhanced.shape[:2]

    # Count detection types
    markers_count = sum(1 for marker in gcp_markers if marker[0] == 'markers')
    numbered_count = sum(1 for marker in gcp_markers if marker[0] == 'marker-numbered')
    total_detections = len(gcp_markers)

    # Add detection indicators
    if total_detections > 0:
        # Determine primary detection type for color coding
        if numbered_count > 0:
            # Numbered markers found - highest priority (red theme)
            overlay = enhanced.copy()
            cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 150), -1)
            enhanced = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)

            cv2.putText(enhanced, f"🔢 NUMBERED MARKERS DETECTED! ({numbered_count} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(enhanced, f"🔢 NUMBERED MARKERS DETECTED! ({numbered_count} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            if markers_count > 0:
                cv2.putText(enhanced, f"📍 Also found {markers_count} general marker(s) for investigation",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        elif markers_count > 0:
            # General markers found (green theme)
            overlay = enhanced.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 150, 0), -1)
            enhanced = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)

            cv2.putText(enhanced, f"📍 MARKERS DETECTED! ({markers_count} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(enhanced, f"📍 MARKERS DETECTED! ({markers_count} found)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Draw big circles around numbered markers (highest priority)
        for class_name, center_x, center_y, bbox_info, confidence in gcp_markers:
            if class_name == 'marker-numbered':
                # Big outer circle for numbered markers
                cv2.circle(enhanced, (center_x, center_y), 100, (0, 0, 255), 5)
                # Inner circle
                cv2.circle(enhanced, (center_x, center_y), 50, (255, 255, 255), 3)
            elif class_name == 'markers':
                # Medium circle for general markers
                cv2.circle(enhanced, (center_x, center_y), 80, (0, 255, 0), 4)

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

def test_gcp_yolo_detection(source: Union[str, int] = 0,
                           display: bool = True,
                           save_results: bool = True,
                           duration: float = 0,
                           video_delay: float = 0.1,
                           model_path: str = "models/best-gcp.pt",
                           confidence: float = 0.5,
                           numbered_confidence: float = 0.4,
                           imgsz: int = 160) -> bool:
    """
    Test GCP YOLO detection with layered detection system.

    Args:
        source: Camera ID (int), video file path, or image file path
        display: Whether to display the detection results
        save_results: Whether to save detection results
        duration: Duration for camera/video (0 = until 'q' pressed)
        video_delay: Delay between frames for video/camera (seconds)
        model_path: Path to GCP YOLO model file
        confidence: Confidence threshold for general markers
        numbered_confidence: Lower confidence threshold for numbered markers
        imgsz: Model input image size

    Returns:
        True if test completed successfully
    """
    try:
        logging.info(f"Starting GCP YOLO detection test with layered detection")
        logging.info(f"Source: {source}")
        logging.info(f"General marker confidence: {confidence}")
        logging.info(f"Numbered marker confidence: {numbered_confidence}")

        detector = GCPYOLODetector(
            model_path=model_path,
            confidence_threshold=confidence,
            numbered_confidence_threshold=numbered_confidence,
            imgsz=imgsz
        )

        # Determine source type
        if isinstance(source, int):
            # Camera input
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logging.error(f"Failed to open camera {source}")
                return False

            # Set camera properties for drone use
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

            logging.info(f"Detected {len(gcp_markers)} GCP markers in image")
            markers_count = sum(1 for marker in gcp_markers if marker[0] == 'markers')
            numbered_count = sum(1 for marker in gcp_markers if marker[0] == 'marker-numbered')

            for i, (class_name, x, y, bbox_info, confidence) in enumerate(gcp_markers):
                logging.info(f"GCP {i+1}: {class_name} at ({x}, {y}), Confidence: {confidence:.3f}")

            if display:
                cv2.imshow("GCP YOLO Detection Results", enhanced_image)
                print("\n" + "="*60)
                print("📸 GCP IMAGE DETECTION RESULTS (LAYERED)")
                print("="*60)
                if len(gcp_markers) > 0:
                    print(f"🎯 GCP MARKERS FOUND: {len(gcp_markers)}")
                    print(f"   📍 General markers: {markers_count}")
                    print(f"   🔢 Numbered markers: {numbered_count}")
                    for i, (class_name, x, y, bbox_info, confidence) in enumerate(gcp_markers):
                        print(f"     {class_name} {i+1}: Position ({x}, {y}), Confidence: {confidence:.1%}")
                else:
                    print("❌ NO GCP MARKERS DETECTED")
                print("="*60)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_results:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gcp_layered_detection_{timestamp}.jpg"
                cv2.imwrite(filename, enhanced_image)
                logging.info(f"Results saved to: {filename}")

            return True

        # Process video or camera feed
        else:
            total_frames = 0
            detection_frames = 0
            markers_detected = 0
            numbered_detected = 0
            start_time = time.time()

            if source_type == "video":
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"Video: {total_video_frames} frames @ {video_fps} FPS")

            logging.info(f"Processing {source_type}. Press 'q' to quit, 's' to save frame")
            print("\n" + "="*80)
            print(f"🎥 {source_type.upper()} GCP LAYERED DETECTION")
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
                if gcp_markers:
                    detection_frames += 1
                    frame_markers = sum(1 for marker in gcp_markers if marker[0] == 'markers')
                    frame_numbered = sum(1 for marker in gcp_markers if marker[0] == 'marker-numbered')

                    markers_detected += frame_markers
                    numbered_detected += frame_numbered

                    print(f"🎯 Frame {total_frames}: {len(gcp_markers)} GCP marker(s) detected!")
                    if frame_numbered > 0:
                        print(f"   🔢 NUMBERED MARKERS: {frame_numbered}")
                    if frame_markers > 0:
                        print(f"   📍 GENERAL MARKERS: {frame_markers}")

                    for i, (class_name, x, y, bbox_info, confidence) in enumerate(gcp_markers):
                        print(f"     → {class_name} at ({x}, {y}), confidence: {confidence:.3f}")

                elif total_frames % 30 == 0:  # Log every 30 frames when no detection
                    print(f"🔍 Frame {total_frames}: Searching for GCP markers...")

                # Display results
                if display:
                    cv2.imshow("GCP YOLO Layered Detection", enhanced_image)

                    key = cv2.waitKey(int(video_delay * 1000)) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_results:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"gcp_layered_frame_{total_frames}_{timestamp}.jpg"
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
            print("📊 GCP LAYERED DETECTION SUMMARY")
            print("="*80)
            print(f"Total frames processed: {total_frames}")
            print(f"Frames with GCP markers: {detection_frames}")
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"Processing time: {total_time:.1f}s")
            print(f"Average FPS: {stats.get('avg_fps', 0):.1f}")
            print(f"Average inference time: {stats.get('avg_inference_ms', 0):.1f}ms")
            print(f"Layered detections: {stats.get('layered_detections', 0)}")

            # Detection breakdown
            print(f"\nDetection Breakdown:")
            print(f"  📍 General markers detected: {markers_detected}")
            print(f"  🔢 Numbered markers detected: {numbered_detected}")

            if detection_frames > 0:
                print(f"🎯 SUCCESS: GCP markers detected in {detection_frames} frames!")
                if numbered_detected > 0:
                    print(f"🔢 NUMBERED MARKERS FOUND: {numbered_detected} total detections!")
                if markers_detected > 0:
                    print(f"📍 GENERAL MARKERS FOUND: {markers_detected} total detections!")
            else:
                print("❌ NO GCP MARKERS DETECTED in any frame")
            print("="*80)

            return True

    except Exception as e:
        logging.error(f"Error during GCP YOLO detection test: {str(e)}")
        return False

def create_gcp_yolo_detector(model_path="models/best-gcp.pt", confidence=0.5,
                            numbered_confidence=0.4, imgsz=160):
    """
    Factory function to create a GCP YOLO detector with layered detection.

    Returns:
        GCPYOLODetector instance
    """
    return GCPYOLODetector(model_path, confidence, numbered_confidence, imgsz)
