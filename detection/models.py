# """
# Detection Models Module
# ---------------------
# Functions for loading and utilizing object detection models.
# """

# import logging
# import torch
# import os
# from ultralytics import YOLO
# import cv2
# import time
# import numpy as np

# def load_detection_model(model_path):
#     """
#     Load a YOLO object detection model.

#     Args:
#         model_path: Path to the model file

#     Returns:
#         Loaded model or None if loading failed
#     """
#     try:
#         # Verify model file exists
#         if not os.path.exists(model_path):
#             logging.error(f"Model file not found: {model_path}")
#             return None

#         # Determine device (GPU if available, otherwise CPU)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         logging.info(f"Loading model on device: {device}")

#         # Load model
#         model = YOLO(model_path)
#         model.to(device)

#         logging.info(f"Model loaded successfully from {model_path}")
#         return model
#     except Exception as e:
#         logging.error(f"Error loading detection model: {str(e)}")
#         return None

# def test_detection_model(model, test_source="0", threshold=0.5, duration=10):
#     """
#     Test the detection model using a camera feed or test image/video.

#     Args:
#         model: The loaded detection model
#         test_source: Source for testing (0 for webcam, or path to image/video)
#         threshold: Detection confidence threshold
#         duration: Test duration in seconds (for webcam only)

#     Returns:
#         True if test was successful, False otherwise
#     """
#     if model is None:
#         logging.error("No model provided for testing")
#         return False

#     try:
#         logging.info(f"Testing detection model on source: {test_source}")

#         # Create debug frames directory if it doesn't exist
#         debug_dir = "debug_frames"
#         if not os.path.exists(debug_dir):
#             os.makedirs(debug_dir)

#         # For webcam, test for a limited duration
#         if test_source == "0" or test_source == 0:
#             cap = cv2.VideoCapture(0)
#             if not cap.isOpened():
#                 logging.error("Cannot open camera")
#                 return False

#             start_time = time.time()
#             frame_count = 0
#             detection_count = 0

#             while time.time() - start_time < duration:
#                 ret, frame = cap.read()
#                 if not ret:
#                     logging.error("Failed to capture frame")
#                     break

#                 # Run detection on frame
#                 results = model.predict(
#                     source=frame,
#                     conf=threshold,
#                     show=True,
#                     save=False
#                 )

#                 # Count detected objects
#                 if len(results) > 0 and len(results[0].boxes) > 0:
#                     detection_count += 1

#                     # Draw bounding boxes on the frame
#                     result_frame = results[0].plot()

#                     # Save frame with detections
#                     cv2.imwrite(f"{debug_dir}/detection_{frame_count}.jpg", result_frame)

#                     # Log detection details
#                     for box in results[0].boxes:
#                         if hasattr(box, 'cls') and hasattr(box, 'conf'):
#                             class_id = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
#                             confidence = float(box.conf[0]) if hasattr(box.conf, '__getitem__') else float(box.conf)
#                             class_name = results[0].names[class_id] if hasattr(results[0], 'names') else f"Class {class_id}"
#                             logging.info(f"Detected {class_name} with confidence {confidence:.2f}")

#                 frame_count += 1

#                 # Process key presses
#                 if cv2.waitKey(1) == ord('q'):
#                     break

#             # Clean up
#             cap.release()
#             cv2.destroyAllWindows()

#             logging.info(f"Test completed. Processed {frame_count} frames, detected objects in {detection_count} frames")
#             return True

#         else:
#             # For image or video file
#             results = model.predict(
#                 source=test_source,
#                 conf=threshold,
#                 show=True,
#                 save=True,
#                 save_txt=True,
#                 save_conf=True,
#                 project=debug_dir,
#                 name="test_results"
#             )

#             # Count detections
#             detection_count = sum(1 for r in results if len(r.boxes) > 0)

#             # Log detection details
#             for i, result in enumerate(results):
#                 if len(result.boxes) > 0:
#                     for box in result.boxes:
#                         if hasattr(box, 'cls') and hasattr(box, 'conf'):
#                             class_id = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
#                             confidence = float(box.conf[0]) if hasattr(box.conf, '__getitem__') else float(box.conf)
#                             class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
#                             logging.info(f"Frame {i}: Detected {class_name} with confidence {confidence:.2f}")

#             logging.info(f"Test completed. Detected objects in {detection_count} frames")
#             return True

#     except Exception as e:
#         logging.error(f"Error testing detection model: {str(e)}")
#         return False

# def run_detection(model, source, threshold=0.5, classes=None, save_results=False, output_dir="detection_results"):
#     """
#     Run object detection on a source (camera, image, video).

#     Args:
#         model: The loaded detection model
#         source: Source for detection (0 for webcam, or path to image/video)
#         threshold: Detection confidence threshold
#         classes: List of class IDs to detect, None for all classes
#         save_results: Whether to save results to disk
#         output_dir: Directory to save results

#     Returns:
#         Generator yielding detection results
#     """
#     if model is None:
#         logging.error("No model provided for detection")
#         return None

#     try:
#         # Configure result saving if enabled
#         save_args = {}
#         if save_results:
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#             save_args = {
#                 'save': True,
#                 'save_txt': True,
#                 'save_conf': True,
#                 'project': output_dir,
#                 'name': f"detection_{time.strftime('%Y%m%d_%H%M%S')}"
#             }

#         # Run detection
#         results = model.predict(
#             source=source,
#             conf=threshold,
#             classes=classes,
#             stream=True,  # Enable streaming for real-time processing
#             **save_args
#         )

#         # Return results generator
#         return results

#     except Exception as e:
#         logging.error(f"Error running detection: {str(e)}")
#         return None

# def process_detection_results(results, frame=None, display=False):
#     """
#     Process detection results for a frame.

#     Args:
#         results: Detection results from model.predict
#         frame: Original frame (if available)
#         display: Whether to display the frame with detections

#     Returns:
#         List of detected objects with their details
#     """
#     try:
#         detections = []

#         # Process results for this frame
#         for result in results:
#             # Skip empty results
#             if not hasattr(result, 'boxes') or len(result.boxes) == 0:
#                 continue

#             # Get the original image
#             img = result.orig_img if frame is None else frame.copy()

#             # Process each detection
#             for i, box in enumerate(result.boxes):
#                 # Get box coordinates
#                 if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 elif hasattr(box, 'xywh') and len(box.xywh) > 0:
#                     x, y, w, h = map(int, box.xywh[0])
#                     x1, y1 = x - w//2, y - h//2
#                     x2, y2 = x + w//2, y + h//2
#                 else:
#                     logging.warning("Box coordinates not found in result")
#                     continue

#                 # Get confidence and class
#                 if hasattr(box, 'conf') and len(box.conf) > 0:
#                     confidence = float(box.conf[0])
#                 else:
#                     confidence = 0.0

#                 if hasattr(box, 'cls') and len(box.cls) > 0:
#                     class_id = int(box.cls[0])
#                     class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
#                 else:
#                     class_id = -1
#                     class_name = "Unknown"

#                 # Calculate center point
#                 center_x = int((x1 + x2) / 2)
#                 center_y = int((y1 + y2) / 2)

#                 # Create detection object
#                 detection = {
#                     'id': i,
#                     'class_id': class_id,
#                     'class_name': class_name,
#                     'confidence': confidence,
#                     'bbox': (x1, y1, x2, y2),
#                     'center': (center_x, center_y),
#                     'width': x2 - x1,
#                     'height': y2 - y1,
#                     'area': (x2 - x1) * (y2 - y1)
#                 }

#                 detections.append(detection)

#                 # Draw on image if display is enabled
#                 if display and img is not None:
#                     # Draw bounding box
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # Draw center point
#                     cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

#                     # Draw label
#                     label = f"{class_name} {confidence:.2f}"
#                     cv2.putText(
#                         img, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#                     )

#             # Display the image
#             if display and img is not None:
#                 cv2.imshow("Detections", img)
#                 cv2.waitKey(1)

#         return detections

#     except Exception as e:
#         logging.error(f"Error processing detection results: {str(e)}")
#         return []

# def save_detection_image(image, detections, output_dir="detection_results", filename=None):
#     """
#     Save an image with detection boxes drawn on it.

#     Args:
#         image: The original image
#         detections: List of detection dictionaries from process_detection_results
#         output_dir: Directory to save the image
#         filename: Filename (if None, use timestamp)

#     Returns:
#         Path to saved file or None if save failed
#     """
#     if image is None or len(detections) == 0:
#         logging.error("Missing image or detections")
#         return None

#     try:
#         # Create output directory if it doesn't exist
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         # Generate filename based on timestamp if not provided
#         if filename is None:
#             filename = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"

#         # Full path to save the image
#         file_path = os.path.join(output_dir, filename)

#         # Create a copy of the image to draw on
#         result_image = image.copy()

#         # Draw each detection
#         for detection in detections:
#             # Extract detection info
#             x1, y1, x2, y2 = detection['bbox']
#             class_name = detection['class_name']
#             confidence = detection['confidence']
#             center_x, center_y = detection['center']

#             # Draw bounding box
#             cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Draw center point
#             cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)

#             # Draw label
#             label = f"{class_name} {confidence:.2f}"
#             cv2.putText(
#                 result_image, label, (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#             )

#         # Save the image
#         cv2.imwrite(file_path, result_image)
#         logging.info(f"Detection image saved to {file_path}")

#         return file_path
#     except Exception as e:
#         logging.error(f"Error saving detection image: {str(e)}")
#         return None
