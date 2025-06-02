# detection/bullseye_detector.py - UPDATED TO USE YOLO MODEL
"""
Bullseye Detection Module using YOLO Model
------------------------------------------
Updated to use trained YOLO model instead of OpenCV pattern matching.
Optimized for drone competition use with Jetson Orin Nano.
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import List, Tuple, Optional, Union
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available - install ultralytics: pip install ultralytics")

class BullseyeDetector:
    """YOLO-based bullseye detector for drone use"""

    def __init__(self, model_path="models/best.pt", confidence_threshold=0.5, imgsz=160):
        """
        Initialize the bullseye detector with YOLO model.

        Args:
            model_path: Path to the YOLO model file (best.pt or best.engine)
            confidence_threshold: Minimum confidence for detections
            imgsz: Input image size for the model
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")

        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.imgsz = imgsz

        # Try to load the model
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            # Try common locations
            alt_paths = [
                "best.pt",
                "models/best.pt",
                "best.engine",
                "models/best.engine"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    self.model_path = alt_path
                    logging.info(f"Found model at: {alt_path}")
                    break
            else:
                raise FileNotFoundError(f"Could not find model file. Tried: {model_path}, {alt_paths}")

        logging.info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.detections_count = 0

    def detect_bullseyes_in_frame(self, frame):
        """
        Detect bullseyes in a single frame using YOLO model.

        Args:
            frame: Input BGR image

        Returns:
            Tuple of (bullseyes, debug_image)
            bullseyes: List of (center_x, center_y, bbox_info, confidence)
            debug_image: Annotated image showing detections
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
            bullseyes = []
            debug_image = frame.copy()

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # Calculate center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Create bbox info dictionary
                        bbox_info = {
                            'bbox': (x1, y1, x2, y2),
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'area': (x2 - x1) * (y2 - y1)
                        }

                        # Add to bullseyes list (format compatible with old system)
                        bullseyes.append((center_x, center_y, bbox_info, confidence))
                        self.detections_count += 1

                        # Draw on debug image
                        self._draw_detection(debug_image, x1, y1, x2, y2, center_x, center_y, confidence)

            # Add frame info to debug image
            self._add_frame_info(debug_image, len(bullseyes), inference_time)

            return bullseyes, debug_image

        except Exception as e:
            logging.error(f"Error in YOLO bullseye detection: {str(e)}")
            return [], frame

    def _draw_detection(self, image, x1, y1, x2, y2, center_x, center_y, confidence):
        """Draw detection on image"""
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

        # Draw crosshair at center
        cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 0), 2)
        cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 0), 2)

        # Add confidence label
        label = f"Bullseye: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def _add_frame_info(self, image, num_detections, inference_time):
        """Add frame information overlay"""
        # Calculate FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Add frame info
        info_text = f"FPS: {fps:.1f} | Detections: {num_detections} | Frame: {self.frame_count}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Add crosshair at frame center for drone alignment reference
        h, w = image.shape[:2]
        cv2.line(image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 0, 255), 2)
        cv2.line(image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 0, 255), 2)
        cv2.circle(image, (w//2, h//2), 3, (0, 0, 255), -1)

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
            'avg_inference_ms': avg_inference_ms
        }

def create_enhanced_detection_display(original_frame, bullseyes, debug_image):
    """
    Create enhanced display with big indicators for detections.
    Updated to work with YOLO detection format.
    """
    enhanced = debug_image.copy()
    height, width = enhanced.shape[:2]

    # Add big detection indicators
    if len(bullseyes) > 0:
        # Green background overlay for success
        overlay = enhanced.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 150, 0), -1)
        enhanced = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)

        # Big success text
        cv2.putText(enhanced, f"🎯 BULLSEYE DETECTED! ({len(bullseyes)} found)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(enhanced, f"🎯 BULLSEYE DETECTED! ({len(bullseyes)} found)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Draw big circles around detections
        for i, (center_x, center_y, bbox_info, confidence) in enumerate(bullseyes):
            # Big outer circle
            cv2.circle(enhanced, (center_x, center_y), 80, (0, 255, 0), 5)
            # Confidence text
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(enhanced, conf_text, (center_x - 60, center_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # Red background overlay for no detection
        overlay = enhanced.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 150), -1)
        enhanced = cv2.addWeighted(enhanced, 0.8, overlay, 0.2, 0)

        # No detection text
        cv2.putText(enhanced, "🔍 NO BULLSEYES DETECTED",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(enhanced, "🔍 NO BULLSEYES DETECTED",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return enhanced

def test_bullseye_detection(source: Union[str, int] = 0,
                          display: bool = True,
                          save_results: bool = True,
                          duration: float = 0,
                          video_delay: float = 0.1,
                          model_path: str = "models/best.pt",
                          confidence: float = 0.5,
                          imgsz: int = 160) -> bool:
    """
    Test YOLO-based bullseye detection on camera feed, video file, or image.

    Args:
        source: Camera ID (int), video file path, or image file path
        display: Whether to display the detection results
        save_results: Whether to save detection results
        duration: Duration for camera/video (0 = until 'q' pressed)
        video_delay: Delay between frames for video/camera (seconds)
        model_path: Path to YOLO model file
        confidence: Confidence threshold for detections
        imgsz: Model input image size

    Returns:
        True if test completed successfully
    """
    try:
        logging.info(f"Starting YOLO bullseye detection test with source: {source}")

        detector = BullseyeDetector(
            model_path=model_path,
            confidence_threshold=confidence,
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
            bullseyes, debug_image = detector.detect_bullseyes_in_frame(image)

            # Create enhanced display image
            enhanced_image = create_enhanced_detection_display(image, bullseyes, debug_image)

            logging.info(f"Detected {len(bullseyes)} bullseyes in image")
            for i, (x, y, bbox_info, confidence) in enumerate(bullseyes):
                logging.info(f"Bullseye {i+1}: Center ({x}, {y}), Confidence: {confidence:.3f}")

            if display:
                cv2.imshow("YOLO Bullseye Detection Results", enhanced_image)
                print("\n" + "="*60)
                print("📸 YOLO IMAGE DETECTION RESULTS")
                print("="*60)
                if len(bullseyes) > 0:
                    print(f"🎯 BULLSEYES FOUND: {len(bullseyes)}")
                    for i, (x, y, bbox_info, confidence) in enumerate(bullseyes):
                        print(f"   Bullseye {i+1}: Position ({x}, {y}), Confidence: {confidence:.1%}")
                else:
                    print("❌ NO BULLSEYES DETECTED")
                print("="*60)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_results:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"yolo_bullseye_detection_{timestamp}.jpg"
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
            print(f"🎥 {source_type.upper()} YOLO BULLSEYE DETECTION")
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

                # Detect bullseyes
                bullseyes, debug_image = detector.detect_bullseyes_in_frame(frame)

                # Create enhanced display
                enhanced_image = create_enhanced_detection_display(frame, bullseyes, debug_image)

                # Track detections
                if bullseyes:
                    detection_frames += 1
                    print(f"🎯 Frame {total_frames}: {len(bullseyes)} bullseye(s) detected!")
                    for i, (x, y, bbox_info, confidence) in enumerate(bullseyes):
                        print(f"   → Bullseye {i+1} at ({x}, {y}), confidence: {confidence:.3f}")
                elif total_frames % 30 == 0:  # Log every 30 frames when no detection
                    print(f"🔍 Frame {total_frames}: Searching for bullseyes...")

                # Display results
                if display:
                    cv2.imshow("YOLO Bullseye Detection", enhanced_image)

                    key = cv2.waitKey(int(video_delay * 1000)) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and save_results:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"yolo_bullseye_frame_{total_frames}_{timestamp}.jpg"
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
            print("📊 YOLO DETECTION SUMMARY")
            print("="*80)
            print(f"Total frames processed: {total_frames}")
            print(f"Frames with bullseyes: {detection_frames}")
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"Processing time: {total_time:.1f}s")
            print(f"Average FPS: {stats.get('avg_fps', 0):.1f}")
            print(f"Average inference time: {stats.get('avg_inference_ms', 0):.1f}ms")

            if detection_frames > 0:
                print(f"🎯 SUCCESS: Bullseyes detected in {detection_frames} frames!")
            else:
                print("❌ NO BULLSEYES DETECTED in any frame")
            print("="*80)

            return True

    except Exception as e:
        logging.error(f"Error during YOLO bullseye detection test: {str(e)}")
        return False

def create_bullseye_detector(model_path="models/best.pt", confidence=0.5, imgsz=160):
    """
    Factory function to create a YOLO bullseye detector.

    Returns:
        BullseyeDetector instance
    """
    return BullseyeDetector(model_path, confidence, imgsz)
