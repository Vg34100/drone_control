# test_position_tracking.py - NEW FUNCTION
"""
GPS Position Tracking Diagnostic Test
------------------------------------
This script tests GPS position tracking while the drone is on the ground.
You can move the drone around manually or walk around with it to test
position data reliability and identify when/why position data drops out.
"""

import time
import logging
import math
from pymavlink import mavutil
from datetime import datetime
import json

def setup_logging():
    """Set up logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('position_test.log'),
            logging.StreamHandler()
        ]
    )

def connect_to_vehicle(connection_string="tcp:127.0.0.1:5761"):
    """Connect to the vehicle"""
    try:
        logging.info(f"Connecting to vehicle at {connection_string}")
        vehicle = mavutil.mavlink_connection(connection_string)
        vehicle.wait_heartbeat()
        logging.info(f"Connected to vehicle (system: {vehicle.target_system}, component: {vehicle.target_component})")
        return vehicle
    except Exception as e:
        logging.error(f"Failed to connect: {str(e)}")
        return None

def request_position_streams(vehicle, rate_hz=10):
    """Request position data streams at specified rate"""
    try:
        # Request GLOBAL_POSITION_INT messages
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            int(1000000 / rate_hz),  # Interval in microseconds
            0, 0, 0, 0, 0
        )

        # Also request GPS_RAW_INT for GPS health monitoring
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
            int(1000000 / rate_hz),
            0, 0, 0, 0, 0
        )

        # Request VFR_HUD for additional data
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
            int(1000000 / rate_hz),
            0, 0, 0, 0, 0
        )

        # Legacy method as backup
        vehicle.mav.request_data_stream_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            rate_hz,
            1
        )

        logging.info(f"Requested position streams at {rate_hz} Hz")
        return True

    except Exception as e:
        logging.error(f"Failed to request position streams: {str(e)}")
        return False

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates"""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    # Haversine formula
    R = 6371000  # Earth radius in meters
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat_rad = math.radians(lat2 - lat1)
    dlon_rad = math.radians(lon2 - lon1)

    a = (math.sin(dlat_rad/2) * math.sin(dlat_rad/2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(dlon_rad/2) * math.sin(dlon_rad/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance

def calculate_speed(distance, time_diff):
    """Calculate speed from distance and time"""
    if distance is None or time_diff <= 0:
        return None
    return distance / time_diff

def run_position_tracking_test(vehicle, duration=300, rate_hz=10):
    """
    Run comprehensive position tracking test

    Args:
        vehicle: Connected mavlink vehicle
        duration: Test duration in seconds (300 = 5 minutes)
        rate_hz: Requested update rate in Hz
    """

    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info("="*80)
        logging.info("STARTING GPS POSITION TRACKING TEST")
        logging.info("="*80)
        logging.info(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
        logging.info(f"Requested rate: {rate_hz} Hz")
        logging.info("You can now move the drone around to test position tracking")
        logging.info("Press Ctrl+C to stop early")
        logging.info("="*80)

        # Request position streams
        if not request_position_streams(vehicle, rate_hz):
            return False

        # Initialize tracking variables
        start_time = time.time()
        last_position_time = None
        last_lat = None
        last_lon = None
        last_alt = None

        # Statistics
        total_messages = 0
        position_messages = 0
        gps_messages = 0
        max_gap = 0
        gaps_over_1s = 0
        total_distance = 0
        max_speed = 0

        # GPS health tracking
        gps_fix_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Extended GPS fix types
        satellite_counts = []

        # Message timing analysis
        message_gaps = []
        expected_interval = 1.0 / rate_hz

        print("\n" + "="*100)
        print("REAL-TIME POSITION TRACKING")
        print("="*100)
        print("Time     | Latitude    | Longitude   | Altitude | Sats | Fix | Speed  | Gap   | Distance")
        print("-"*100)

        try:
            while time.time() - start_time < duration:
                current_time = time.time()

                # Process all available messages
                message_found = False
                while True:
                    msg = vehicle.recv_match(blocking=False, timeout=0.01)
                    if not msg:
                        break

                    total_messages += 1
                    message_found = True
                    msg_type = msg.get_type()

                    # Process GLOBAL_POSITION_INT messages
                    if msg_type == "GLOBAL_POSITION_INT":
                        position_messages += 1

                        # Extract position data
                        current_lat = msg.lat / 1e7
                        current_lon = msg.lon / 1e7
                        current_alt = msg.relative_alt / 1000.0

                        # Calculate timing gap
                        gap = 0
                        if last_position_time:
                            gap = current_time - last_position_time
                            message_gaps.append(gap)
                            max_gap = max(max_gap, gap)
                            if gap > 1.0:
                                gaps_over_1s += 1

                        # Calculate distance and speed
                        distance = 0
                        speed = 0
                        if last_lat is not None and last_lon is not None:
                            distance = calculate_distance(last_lat, last_lon, current_lat, current_lon)
                            if distance:
                                total_distance += distance
                                if gap > 0:
                                    speed = calculate_speed(distance, gap)
                                    if speed:
                                        max_speed = max(max_speed, speed)

                        # Display current position
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"{timestamp} | {current_lat:>11.7f} | {current_lon:>11.7f} | {current_alt:>8.2f} | ---- | --- | {speed:>6.2f} | {gap:>5.2f} | {distance:>8.2f}")

                        # Update tracking variables
                        last_position_time = current_time
                        last_lat = current_lat
                        last_lon = current_lon
                        last_alt = current_alt

                    # Process GPS_RAW_INT messages for GPS health
                    elif msg_type == "GPS_RAW_INT":
                        gps_messages += 1

                        fix_type = msg.fix_type
                        satellites = msg.satellites_visible

                        # Track GPS statistics
                        if fix_type in gps_fix_counts:
                            gps_fix_counts[fix_type] += 1
                        satellite_counts.append(satellites)

                        # Update display with GPS info
                        fix_names = {0: "NON", 1: "NOF", 2: "2D ", 3: "3D ", 4: "DGP", 5: "RTK", 6: "FLT"}
                        fix_name = fix_names.get(fix_type, f"{fix_type:3d}")

                        # Overwrite last line with GPS info
                        if position_messages > 0:  # Only if we have position data
                            print(f"\r{timestamp} | {current_lat:>11.7f} | {current_lon:>11.7f} | {current_alt:>8.2f} | {satellites:>4d} | {fix_name} | {speed:>6.2f} | {gap:>5.2f} | {distance:>8.2f}", end="")

                # Check for message timeout
                if not message_found and last_position_time:
                    gap = current_time - last_position_time
                    if gap > 2.0:  # 2 second timeout warning
                        print(f"\n⚠️  WARNING: No position data for {gap:.1f} seconds!", end="")

                # Brief sleep to prevent 100% CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n\nTest stopped by user after {time.time() - start_time:.1f} seconds")

        # Calculate final statistics
        test_duration = time.time() - start_time
        avg_gap = sum(message_gaps) / len(message_gaps) if message_gaps else 0
        actual_rate = position_messages / test_duration if test_duration > 0 else 0
        avg_satellites = sum(satellite_counts) / len(satellite_counts) if satellite_counts else 0

        # Print comprehensive summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Test Duration: {test_duration:.1f} seconds")
        print(f"Requested Rate: {rate_hz} Hz")
        print(f"Actual Rate: {actual_rate:.2f} Hz")
        print(f"Rate Efficiency: {(actual_rate/rate_hz*100):.1f}%")
        print()
        print("MESSAGE STATISTICS:")
        print(f"  Total Messages: {total_messages}")
        print(f"  Position Messages: {position_messages}")
        print(f"  GPS Health Messages: {gps_messages}")
        print()
        print("TIMING ANALYSIS:")
        print(f"  Expected Interval: {expected_interval:.3f}s")
        print(f"  Average Gap: {avg_gap:.3f}s")
        print(f"  Maximum Gap: {max_gap:.3f}s")
        print(f"  Gaps > 1 second: {gaps_over_1s}")
        print()
        print("GPS HEALTH:")
        print(f"  Average Satellites: {avg_satellites:.1f}")
        print(f"  GPS Fix Distribution:")
        fix_names = {0: "No GPS", 1: "No Fix", 2: "2D Fix", 3: "3D Fix", 4: "3D DGPS", 5: "RTK Float", 6: "RTK Fixed"}
        for fix_type, count in gps_fix_counts.items():
            if count > 0:  # Only show fix types that occurred
                percentage = (count / gps_messages * 100) if gps_messages > 0 else 0
                print(f"    {fix_names.get(fix_type, f'Fix Type {fix_type}')}: {count} ({percentage:.1f}%)")
        print()
        print("MOVEMENT ANALYSIS:")
        print(f"  Total Distance: {total_distance:.2f} meters")
        print(f"  Maximum Speed: {max_speed:.2f} m/s")
        print(f"  Average Speed: {total_distance/test_duration:.2f} m/s")

        # Identify potential issues
        print("\n" + "="*80)
        print("POTENTIAL ISSUES DETECTED:")
        print("="*80)

        issues_found = False

        if actual_rate < rate_hz * 0.8:
            print(f"❌ LOW MESSAGE RATE: Getting {actual_rate:.1f} Hz instead of {rate_hz} Hz")
            issues_found = True

        if max_gap > 2.0:
            print(f"❌ LARGE MESSAGE GAPS: Maximum gap of {max_gap:.1f} seconds detected")
            issues_found = True

        if gaps_over_1s > 5:
            print(f"❌ FREQUENT GAPS: {gaps_over_1s} gaps over 1 second detected")
            issues_found = True

        if avg_satellites < 8:
            print(f"⚠️  LOW SATELLITE COUNT: Average of {avg_satellites:.1f} satellites")
            issues_found = True

        # Check for good GPS fixes (3D or better)
        good_gps_fixes = gps_fix_counts.get(3, 0) + gps_fix_counts.get(4, 0) + gps_fix_counts.get(5, 0) + gps_fix_counts.get(6, 0)
        good_gps_percentage = (good_gps_fixes / gps_messages * 100) if gps_messages > 0 else 0
        if good_gps_percentage < 95:
            print(f"⚠️  POOR GPS FIX: Only {good_gps_percentage:.1f}% good GPS fixes (3D/DGPS/RTK)")
            issues_found = True
        else:
            print(f"✅ EXCELLENT GPS: {good_gps_percentage:.1f}% good GPS fixes")

        # Check message rate efficiency
        rate_efficiency = (actual_rate / rate_hz * 100) if rate_hz > 0 else 0
        if rate_efficiency < 80:
            print(f"❌ LOW MESSAGE RATE EFFICIENCY: {rate_efficiency:.1f}% of requested rate")
            print(f"   This suggests autopilot message throttling or connection issues")
            issues_found = True

        if not issues_found:
            print("✅ No major issues detected!")

        # Save detailed log
        log_data = {
            'test_duration': test_duration,
            'requested_rate': rate_hz,
            'actual_rate': actual_rate,
            'total_messages': total_messages,
            'position_messages': position_messages,
            'message_gaps': message_gaps,
            'gps_fix_counts': gps_fix_counts,
            'satellite_counts': satellite_counts,
            'total_distance': total_distance,
            'max_speed': max_speed
        }

        with open('position_test_results.json', 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\nDetailed results saved to position_test_results.json")
        print("="*80)

        return True

    except Exception as e:
        logging.error(f"Error during position tracking test: {str(e)}")
        return False

def main():
    """Main function to run the position tracking test"""
    setup_logging()

    import argparse
    parser = argparse.ArgumentParser(description="GPS Position Tracking Diagnostic Test")
    parser.add_argument("--connection", default="tcp:127.0.0.1:5761", help="Connection string")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--rate", type=int, default=10, help="Requested update rate in Hz")

    args = parser.parse_args()

    # Connect to vehicle
    vehicle = connect_to_vehicle(args.connection)
    if not vehicle:
        return 1

    try:
        # Run the test
        success = run_position_tracking_test(vehicle, args.duration, args.rate)
        return 0 if success else 1

    finally:
        if vehicle:
            vehicle.close()

if __name__ == "__main__":
    exit(main())
