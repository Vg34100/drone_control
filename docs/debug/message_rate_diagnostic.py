# message_rate_diagnostic.py - NEW FUNCTION
"""
Message Rate Diagnostic Tool
---------------------------
This script specifically diagnoses why you're getting 3.6 Hz instead of 10 Hz
for position messages. It tests different request methods and rates.
"""

import time
import logging
from pymavlink import mavutil
from collections import defaultdict
import statistics

def test_message_rate_methods(connection_string="tcp:127.0.0.1:5761"):
    """Test different methods of requesting position messages"""

    print("=== MESSAGE RATE DIAGNOSTIC ===")
    print("Testing different message request methods...")
    print("=" * 80)

    try:
        # Connect
        print("Connecting to vehicle...")
        vehicle = mavutil.mavlink_connection(connection_string)
        vehicle.wait_heartbeat()
        print(f"Connected to system {vehicle.target_system}, component {vehicle.target_component}")

        # Get autopilot info
        print("\nRequesting autopilot version...")
        vehicle.mav.command_long_send(
            vehicle.target_system, vehicle.target_component,
            mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES, 0,
            1, 0, 0, 0, 0, 0, 0
        )

        # Check what the autopilot reports as capabilities
        version_msg = vehicle.recv_match(type='AUTOPILOT_VERSION', blocking=True, timeout=3)
        if version_msg:
            print(f"Autopilot capabilities: {version_msg.capabilities}")
        else:
            print("No autopilot version received")

        # Test different request rates
        test_rates = [1, 2, 5, 10, 20]

        for requested_rate in test_rates:
            print(f"\n" + "="*50)
            print(f"TESTING REQUESTED RATE: {requested_rate} Hz")
            print("="*50)

            # Clear message buffer
            while vehicle.recv_match(blocking=False):
                pass

            # Method 1: Message interval request
            print(f"Method 1: SET_MESSAGE_INTERVAL for {requested_rate} Hz")
            interval_us = int(1000000 / requested_rate)

            vehicle.mav.command_long_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                interval_us, 0, 0, 0, 0, 0
            )

            # Wait for acknowledgment
            ack = vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
            if ack and ack.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
                ack_result = {
                    0: "ACCEPTED",
                    1: "TEMPORARILY_REJECTED",
                    2: "DENIED",
                    3: "UNSUPPORTED",
                    4: "FAILED"
                }.get(ack.result, f"UNKNOWN({ack.result})")
                print(f"  Command ACK: {ack_result}")
            else:
                print("  No ACK received for message interval command")

            # Method 2: Legacy data stream request
            print(f"Method 2: REQUEST_DATA_STREAM for {requested_rate} Hz")
            vehicle.mav.request_data_stream_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                requested_rate, 1
            )

            # Measure actual rate for 10 seconds
            print(f"Measuring actual rate for 10 seconds...")

            start_time = time.time()
            message_times = []
            test_duration = 10

            print("Time     | Messages | Current Rate | Avg Rate | Gap")
            print("-" * 50)

            while time.time() - start_time < test_duration:
                msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                if msg:
                    msg_time = time.time()
                    message_times.append(msg_time)

                    # Calculate current rate (over last 5 messages)
                    if len(message_times) >= 5:
                        recent_times = message_times[-5:]
                        time_span = recent_times[-1] - recent_times[0]
                        current_rate = 4 / time_span if time_span > 0 else 0
                    else:
                        current_rate = 0

                    # Calculate average rate
                    elapsed = msg_time - start_time
                    avg_rate = len(message_times) / elapsed if elapsed > 0 else 0

                    # Calculate gap from last message
                    gap = message_times[-1] - message_times[-2] if len(message_times) > 1 else 0

                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\r{timestamp} | {len(message_times):>8} | {current_rate:>12.2f} | {avg_rate:>8.2f} | {gap:>5.3f}", end="")

                else:
                    print("\nTimeout waiting for message!")
                    break

            # Calculate final statistics
            if len(message_times) >= 2:
                total_time = message_times[-1] - message_times[0]
                actual_rate = (len(message_times) - 1) / total_time

                # Calculate message intervals
                intervals = [message_times[i] - message_times[i-1] for i in range(1, len(message_times))]
                avg_interval = statistics.mean(intervals)
                std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
                min_interval = min(intervals)
                max_interval = max(intervals)

                print(f"\n\nRESULTS for {requested_rate} Hz request:")
                print(f"  Requested Rate: {requested_rate:.1f} Hz")
                print(f"  Actual Rate: {actual_rate:.2f} Hz")
                print(f"  Efficiency: {(actual_rate/requested_rate*100):.1f}%")
                print(f"  Messages Received: {len(message_times)}")
                print(f"  Average Interval: {avg_interval:.3f}s (should be {1/requested_rate:.3f}s)")
                print(f"  Interval Std Dev: {std_interval:.3f}s")
                print(f"  Min Interval: {min_interval:.3f}s")
                print(f"  Max Interval: {max_interval:.3f}s")

                # Identify the issue
                if actual_rate < requested_rate * 0.8:
                    print(f"  ❌ ISSUE: Getting {actual_rate:.1f} Hz instead of {requested_rate} Hz")
                    if std_interval > 0.1:
                        print(f"  ❌ ISSUE: High interval variability ({std_interval:.3f}s)")
                    if max_interval > (2 / requested_rate):
                        print(f"  ❌ ISSUE: Large gaps detected (max: {max_interval:.3f}s)")
                else:
                    print(f"  ✅ GOOD: Rate is within acceptable range")
            else:
                print(f"\n❌ FAILED: No messages received at {requested_rate} Hz")

            # Stop the stream before next test
            vehicle.mav.command_long_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                0, 0, 0, 0, 0, 0  # 0 = stop
            )

            vehicle.mav.request_data_stream_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                0, 0  # Stop stream
            )

            time.sleep(1)  # Brief pause between tests

        # Test what rates the autopilot actually supports
        print(f"\n" + "="*80)
        print("AUTOPILOT RATE LIMITS TEST")
        print("="*80)
        print("Testing maximum sustainable rate...")

        # Test very high rate to see what autopilot can actually do
        high_rates = [50, 100, 200]

        for high_rate in high_rates:
            print(f"\nTesting {high_rate} Hz request...")

            # Clear buffer
            while vehicle.recv_match(blocking=False):
                pass

            # Request high rate
            interval_us = int(1000000 / high_rate)
            vehicle.mav.command_long_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                interval_us, 0, 0, 0, 0, 0
            )

            # Measure for 5 seconds
            start_time = time.time()
            msg_count = 0

            while time.time() - start_time < 5:
                msg = vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.1)
                if msg:
                    msg_count += 1

            actual_rate = msg_count / 5
            print(f"  Requested: {high_rate} Hz, Actual: {actual_rate:.2f} Hz")

            # Stop stream
            vehicle.mav.command_long_send(
                vehicle.target_system, vehicle.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                0, 0, 0, 0, 0, 0
            )

        print(f"\n" + "="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)

        vehicle.close()
        return True

    except Exception as e:
        print(f"Error during message rate test: {str(e)}")
        return False

if __name__ == "__main__":
    test_message_rate_methods()
