"""
Ground Control Point (GCP) Detection Module
----------------------------------------
Functions for detecting ground control points like squares, triangles,
and X-patterns for drone navigation and landing.
"""

import cv2
import numpy as np
import math
import logging
import os
import time

def detect_squares(frame, min_area=100, max_area=10000, aspect_ratio_range=(0.8, 1.2)):
    """
    Detect squares in the frame.

    Args:
        frame: Input image frame
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        aspect_ratio_range: Tuple of (min, max) aspect ratio for square detection

    Returns:
        List of detected squares, each as (x, y, w, h, contour)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store detected squares
        squares = []

        # Loop through all contours
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Approximate the contour shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Check if it has 4 vertices (potential square)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)

                # Check aspect ratio for squareness
                aspect_ratio = float(w) / h
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    squares.append((x, y, w, h, approx))

        return squares
    except Exception as e:
        logging.error(f"Error detecting squares: {str(e)}")
        return []

def detect_triangles(frame, min_area=100, max_area=10000, tolerance=0.15):
    """
    Detect triangles in the frame.

    Args:
        frame: Input image frame
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        tolerance: Tolerance for side length similarity in equilateral triangles

    Returns:
        List of detected triangles, each as (x, y, w, h, contour)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store detected triangles
        triangles = []

        # Loop through all contours
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Approximate the contour shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Check if it has 3 vertices (triangle)
            if len(approx) == 3:
                # Calculate side lengths
                side1 = np.linalg.norm(approx[0][0] - approx[1][0])
                side2 = np.linalg.norm(approx[1][0] - approx[2][0])
                side3 = np.linalg.norm(approx[2][0] - approx[0][0])

                # Calculate average side length
                avg_side = (side1 + side2 + side3) / 3

                # Check if it's approximately equilateral
                if (abs(side1 - avg_side) / avg_side < tolerance and
                    abs(side2 - avg_side) / avg_side < tolerance and
                    abs(side3 - avg_side) / avg_side < tolerance):

                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    triangles.append((x, y, w, h, approx))

        return triangles
    except Exception as e:
        logging.error(f"Error detecting triangles: {str(e)}")
        return []

def detect_x_pattern(frame, min_area=500, max_area=100000, diagonal_angle_tolerance=15):
    """
    Detect X-patterns in the frame (squares with diagonal lines).

    Args:
        frame: Input image frame
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        diagonal_angle_tolerance: Tolerance for diagonal angles (45±tolerance)

    Returns:
        List of detected X-patterns, each as (x, y, w, h, confidence)
    """
    try:
        # First, detect potential squares
        squares = detect_squares(frame, min_area, max_area)

        # Process each square to check for X-pattern
        x_patterns = []

        for x, y, w, h, approx in squares:
            # Create a mask for this square
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [approx], 0, 255, -1)

            # Get the ROI (Region of Interest)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = cv2.bitwise_and(gray, gray, mask=mask)

            # Skip if ROI is empty
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

            # Count diagonal lines
            diagonal_lines = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # Calculate line angle
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

                    # Check if line is close to diagonal angles (45 or 135 degrees)
                    if ((45 - diagonal_angle_tolerance <= angle <= 45 + diagonal_angle_tolerance) or
                        (135 - diagonal_angle_tolerance <= angle <= 135 + diagonal_angle_tolerance)):
                        diagonal_lines.append((x1, y1, x2, y2, angle))

            # Check quadrant intensity differences as an alternative method
            h_half, w_half = h // 2, w // 2
            quadrant_pattern_detected = False

            if h_half > 0 and w_half > 0:
                # Define quadrant regions
                q1 = roi[y:y+h_half, x:x+w_half]
                q2 = roi[y:y+h_half, x+w_half:x+w]
                q3 = roi[y+h_half:y+h, x:x+w_half]
                q4 = roi[y+h_half:y+h, x+w_half:x+w]

                # Calculate mean intensity for non-zero pixels in each quadrant
                q1_intensity = np.mean(q1[q1 > 0]) if np.any(q1 > 0) else 0
                q2_intensity = np.mean(q2[q2 > 0]) if np.any(q2 > 0) else 0
                q3_intensity = np.mean(q3[q3 > 0]) if np.any(q3 > 0) else 0
                q4_intensity = np.mean(q4[q4 > 0]) if np.any(q4 > 0) else 0

                # Calculate intensity differences
                diagonal1_diff = abs(q1_intensity - q4_intensity)
                diagonal2_diff = abs(q2_intensity - q3_intensity)
                cross_diff = abs((q1_intensity + q4_intensity) / 2 - (q2_intensity + q3_intensity) / 2)

                # If diagonals have similar intensities within each diagonal but
                # different between diagonals, it's likely an X pattern
                quadrant_pattern_detected = (
                    diagonal1_diff < 10 and
                    diagonal2_diff < 10 and
                    cross_diff > 20
                )

            # Check if this is an X-pattern based on diagonal lines and quadrant analysis
            has_enough_diagonals = len(diagonal_lines) >= 2

            if has_enough_diagonals or quadrant_pattern_detected:
                # Calculate confidence score
                confidence = (len(diagonal_lines) * 0.3) + (1 if quadrant_pattern_detected else 0)
                x_patterns.append((x, y, w, h, confidence))

        return x_patterns
    except Exception as e:
        logging.error(f"Error detecting X-patterns: {str(e)}")
        return []

def detect_triangles_in_squares(frame, min_triangles=4):
    """
    Detect squares containing multiple triangles.

    Args:
        frame: Input image frame
        min_triangles: Minimum number of triangles required inside a square

    Returns:
        List of detected patterns, each as (x, y, w, h, num_triangles)
    """
    try:
        # Detect triangles and squares
        triangles = detect_triangles(frame)
        squares = detect_squares(frame)

        # Check which squares contain multiple triangles
        patterns = []

        for sx, sy, sw, sh, s_approx in squares:
            # Count triangles inside this square
            triangles_inside = 0

            for tx, ty, tw, th, t_approx in triangles:
                # Check if triangle center is inside square
                t_center_x = tx + tw/2
                t_center_y = ty + th/2

                if (sx <= t_center_x <= sx + sw and
                    sy <= t_center_y <= sy + sh):
                    triangles_inside += 1

            # If enough triangles are inside the square
            if triangles_inside >= min_triangles:
                patterns.append((sx, sy, sw, sh, triangles_inside))

        return patterns
    except Exception as e:
        logging.error(f"Error detecting triangles in squares: {str(e)}")
        return []

def detect_gcp_markers(frame, save_debug=False, debug_dir="debug_frames"):
    """
    Detect all types of GCP markers in a frame.

    Args:
        frame: Input image frame
        save_debug: Whether to save debug frames to disk
        debug_dir: Directory to save debug frames

    Returns:
        Dictionary of detected markers by type
    """
    try:
        # Create copy for visualization
        display_frame = frame.copy()

        # Detect different types of markers
        squares = detect_squares(frame)
        triangles = detect_triangles(frame)
        x_patterns = detect_x_pattern(frame)
        tri_in_squares = detect_triangles_in_squares(frame)

        # Draw detections on the display frame
        for x, y, w, h, _ in squares:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Square", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for x, y, w, h, _ in triangles:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(display_frame, "Triangle", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for x, y, w, h, conf in x_patterns:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display_frame, f"X-Pattern ({conf:.1f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for x, y, w, h, count in tri_in_squares:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(display_frame, f"{count} triangles", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Save debug frame if requested
        if save_debug:
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{debug_dir}/gcp_detection_{timestamp}.jpg", display_frame)

        # Return all detections
        return {
            'display_frame': display_frame,
            'squares': squares,
            'triangles': triangles,
            'x_patterns': x_patterns,
            'tri_in_squares': tri_in_squares
        }
    except Exception as e:
        logging.error(f"Error detecting GCP markers: {str(e)}")
        return {
            'display_frame': frame,
            'squares': [],
            'triangles': [],
            'x_patterns': [],
            'tri_in_squares': []
        }

def run_gcp_detection(camera_id=0, duration=30, save_detected=True):
    """
    Run continuous GCP detection on a camera feed.

    Args:
        camera_id: Camera ID to use
        duration: Duration in seconds (0 for indefinite)
        save_detected: Whether to save frames with detections

    Returns:
        True if completed successfully, False otherwise
    """
    try:
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Failed to open camera {camera_id}")
            return False

        # Create debug directory
        debug_dir = "debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # Start detection loop
        start_time = time.time()
        frame_count = 0
        detection_count = 0

        logging.info(f"Starting GCP detection for {duration} seconds (0 = indefinite)")

        while True:
            # Check if duration is reached
            if duration > 0 and (time.time() - start_time > duration):
                break

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                break

            frame_count += 1

            # Run detection every 5 frames to reduce processing load
            if frame_count % 5 == 0:
                # Detect GCP markers
                results = detect_gcp_markers(frame, save_debug=False)

                # Check if any markers were detected
                has_detections = (
                    len(results['squares']) > 0 or
                    len(results['triangles']) > 0 or
                    len(results['x_patterns']) > 0 or
                    len(results['tri_in_squares']) > 0
                )

                if has_detections:
                    detection_count += 1

                    # Save frame with detections if requested
                    if save_detected:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"{debug_dir}/gcp_detection_{timestamp}.jpg", results['display_frame'])

                # Display the frame with detections
                cv2.imshow("GCP Detection", results['display_frame'])

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        logging.info(f"GCP detection completed. Processed {frame_count} frames, found detections in {detection_count} frames")
        return True
    except Exception as e:
        logging.error(f"Error during GCP detection: {str(e)}")
        return False
