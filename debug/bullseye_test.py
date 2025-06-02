# FILE: bullseye_video_test.py
# New script for testing bullseye detection on video files
# Optimized for drone competition use with Jetson Orin Nano

import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class BullseyeVideoTester:
    def __init__(self, model_path="models/best.pt", confidence_threshold=0.5, imgsz=160):
        """
        Initialize the bullseye detector

        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
            imgsz: Input image size for the model
        """
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.imgsz = imgsz

        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
        self.detections_count = 0

    def detect_bullseyes(self, frame):
        """
        Detect bullseyes in a single frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            annotated_frame: Frame with bounding boxes drawn
            detections: List of detection dictionaries
        """
        start_time = time.time()

        # Run inference
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)

        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1

        # Process results
        detections = []
        annotated_frame = frame.copy()

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # Store detection info
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    }
                    detections.append(detection)
                    self.detections_count += 1

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw center point
                    center = detection['center']
                    cv2.circle(annotated_frame, center, 5, (255, 0, 0), -1)

                    # Add confidence label
                    label = f"Bullseye: {confidence:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add frame info
        fps = 1.0 / inference_time if inference_time > 0 else 0
        info_text = f"FPS: {fps:.1f} | Detections: {len(detections)} | Frame: {self.frame_count}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add crosshair at frame center for drone alignment reference
        h, w = annotated_frame.shape[:2]
        cv2.line(annotated_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 0, 255), 2)
        cv2.line(annotated_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 0, 255), 2)

        return annotated_frame, detections

    def process_video(self, video_path, output_path=None, display=True, save_video=False):
        """
        Process video file and detect bullseyes

        Args:
            video_path: Path to input video file
            output_path: Path for output video (if save_video=True)
            display: Whether to display video while processing
            save_video: Whether to save annotated video
        """
        print(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video specs: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup video writer if saving
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        all_detections = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect bullseyes
                annotated_frame, detections = self.detect_bullseyes(frame)
                all_detections.extend(detections)

                # Save frame if requested
                if save_video and output_path:
                    out.write(annotated_frame)

                # Display frame
                if display:
                    # Resize for display if frame is too large
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))

                    cv2.imshow('Bullseye Detection', display_frame)

                    # Controls: 'q' to quit, 'p' to pause, space to continue
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)  # Wait for any key

                # Progress indicator
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")

        finally:
            # Cleanup
            cap.release()
            if save_video and output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()

        # Print summary
        self.print_summary(all_detections)

    def process_camera(self, camera_id=0):
        """
        Process live camera feed (for testing with drone camera)

        Args:
            camera_id: Camera device ID (0 for default camera)
        """
        print(f"Starting camera feed from device {camera_id}")
        print("Controls: 'q' to quit, 's' to save screenshot")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        # Set camera properties (adjust for your drone camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        screenshot_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip frame horizontally for better user experience
                frame = cv2.flip(frame, 1)

                # Detect bullseyes
                annotated_frame, detections = self.detect_bullseyes(frame)

                # Display
                cv2.imshow('Live Bullseye Detection', annotated_frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"bullseye_screenshot_{screenshot_count:03d}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                    screenshot_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()

        self.print_summary([])

    def print_summary(self, all_detections):
        """Print performance summary"""
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {self.detections_count}")
        print(f"Average detections per frame: {self.detections_count/max(1, self.frame_count):.2f}")

        if self.total_inference_time > 0:
            avg_fps = self.frame_count / self.total_inference_time
            avg_inference_ms = (self.total_inference_time / self.frame_count) * 1000
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Average inference time: {avg_inference_ms:.1f}ms")

        if all_detections:
            confidences = [d['confidence'] for d in all_detections]
            print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"Average confidence: {np.mean(confidences):.3f}")

        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Test bullseye detection on video')
    parser.add_argument('--model', default='models/best.pt', help='Path to YOLO model')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', action='store_true', help='Use camera instead of video file')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=160, help='Model input size')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--save', action='store_true', help='Save annotated video')

    args = parser.parse_args()

    # Initialize detector
    detector = BullseyeVideoTester(
        model_path=args.model,
        confidence_threshold=args.conf,
        imgsz=args.imgsz
    )

    if args.camera:
        # Live camera mode
        detector.process_camera(args.camera_id)
    elif args.video:
        # Video file mode
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            return

        output_path = args.output
        if args.save and not output_path:
            # Auto-generate output filename
            video_path = Path(args.video)
            output_path = video_path.parent / f"{video_path.stem}_detected{video_path.suffix}"

        detector.process_video(
            video_path=args.video,
            output_path=output_path,
            display=not args.no_display,
            save_video=args.save
        )
    else:
        print("Error: Please specify either --video or --camera")
        parser.print_help()

if __name__ == "__main__":
    main()
