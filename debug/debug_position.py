# debug_position.py - NEW FUNCTION
"""
Quick Position Debug Script
--------------------------
This script quickly tests your position data issue while the drone is on the ground.
Run this to see if you can reproduce the position data dropouts.
"""

import time
import logging
from pymavlink import mavutil

def quick_position_test(connection_string="tcp:127.0.0.1:5761", duration=60):
    """Quick test of position data reliability"""

    print("=== QUICK POSITION DEBUG TEST ===")
    print(f"Duration: {duration} seconds")
    print("Testing position data reliability...")
    print("=" * 50)

    try:
        # Connect
        print("Connecting to vehicle...")
        vehicle = mavutil.mavlink_connection(connection_string)
        vehicle.wait_heartbeat()
        print(f"Connected to system {vehicle.target_system}")

        # Request position data multiple ways
        print("Requesting position streams...")

        # Method 1: Modern message interval
        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            200000, 0, 0, 0, 0, 0  # 5 Hz
        )

        # Method 2: Legacy stream request
        vehicle.mav.request_data_stream_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5, 1
        )

        print("Starting position monitoring...")
        print("Time     | Latitude    | Longitude   | Altitude | Gap   | Status")
        print("-" * 70)

        start_time = time.time()
        last_msg_time = None
        msg_count = 0
        gaps = []

        while time.time() - start_time < duration:
            # Try to get position message
            msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2.0)

            current_time = time.time()

            if msg:
                msg_count += 1
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.relative_alt / 1000.0

                # Calculate gap
                gap = 0
                if last_msg_time:
                    gap = current_time - last_msg_time
                    gaps.append(gap)

                # Status
                status = "OK"
                if gap > 1.0:
                    status = "LONG GAP!"
                elif gap > 0.5:
                    status = "gap"

                timestamp = time.strftime("%H:%M:%S")
                print(f"{timestamp} | {lat:>11.7f} | {lon:>11.7f} | {alt:>8.2f} | {gap:>5.2f} | {status}")

                last_msg_time = current_time

            else:
                # No message received
                gap = current_time - last_msg_time if last_msg_time else 0
                timestamp = time.strftime("%H:%M:%S")
                print(f"{timestamp} | {'NO DATA':>11} | {'NO DATA':>11} | {'NO DATA':>8} | {gap:>5.2f} | TIMEOUT!")

        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"Messages received: {msg_count}")
        print(f"Expected messages: ~{duration * 5}")
        print(f"Success rate: {(msg_count / (duration * 5)) * 100:.1f}%")

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            max_gap = max(gaps)
            print(f"Average gap: {avg_gap:.3f}s")
            print(f"Maximum gap: {max_gap:.3f}s")
            long_gaps = [g for g in gaps if g > 1.0]
            print(f"Gaps > 1s: {len(long_gaps)}")

        vehicle.close()
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    quick_position_test(duration=duration)
