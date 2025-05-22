#!/usr/bin/env python3
"""
Drone Control System - Main Launcher
-----------------------------------
This script serves as the main entry point for all drone operations.
It parses command-line arguments to determine which mission to run.
"""

import argparse
import sys
import logging
import time
from pymavlink import mavutil

# Import modules
from drone.connection import connect_vehicle, close_vehicle
from drone.navigation import set_mode, test_motors
from missions.test_missions import (
    test_connection,
    test_arm,
    test_takeoff,
    test_camera,
    test_detection,
    test_motor
)
from missions.waypoint import (
    mission_waypoint,
    mission_waypoint_detect
)
from missions.delivery import (
    mission_package_delivery,
    mission_package_drop,
    mission_target_localize
)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='drone_mission.log'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Drone Mission Control System")

    # Mission selection argument
    parser.add_argument(
        "mission",
        choices=[
            "test-connection",      # Test connection to the vehicle and runs diagnostics
            "preflight-all",        # Run all preflight checks (connection, arm, motors, and preflight tests)
            "test-arm",             # Test arming the vehicle
            "test-motor",           # Test each motor functionality
            "test-camera",          # Test camera functionality
            "test-detect",

            # Test takeoff scripts
            "test-takeoff",
            "incremental-takeoff",      # New incremental takeoff test
            "diamond-waypoints",

            "waypoint",
            "waypoint-detect",
            "package-delivery",
            "package-drop",
            "target-localize",

            # Fixes
            "fix-mode",             # Fix the mode after RTL if necessary

            "diagnostics",          # New command
            "reset-controller",     # Resets the flight controller

            # Preflight
            "safety-check",             # To run safety checks only
            "orientation-check",        # To check orientation stability
            "position-hold-check",      # To test position holding

            "check-altitude",           # Real-time altitude monitoring

        ],
        help="Mission to execute"
    )

    # Optional arguments
    parser.add_argument(
        "--altitude",
        type=float,
        default=3.0,
        help="Target altitude in meters"
    )
    parser.add_argument(
        "--connection",
        type=str,
        default="tcp:127.0.0.1:5761",
        help="Connection string for the vehicle"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/latest.pt",
        help="Path to the detection model"
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=15.0,
        help="Throttle percentage for motor testing (0-100)"
    )
    parser.add_argument(
    "--increment",
    type=float,
    default=1.0,
    help="Height increment in meters for incremental takeoff test"
    )

    args = parser.parse_args()

    # Connect to vehicle for missions that require it
    vehicle = None
    mission_requires_vehicle = args.mission not in ["test-camera"]

    try:
        # Initialize vehicle connection if needed
        if mission_requires_vehicle:
            logging.info(f"Connecting to vehicle at {args.connection}")
            vehicle = connect_vehicle(args.connection)
            if not vehicle:
                logging.error("Failed to connect to vehicle")
                return 1

        # Execute the selected mission
        if args.mission == "test-connection":
            success = test_connection(vehicle)
        elif args.mission == "test-arm":
            success = test_arm(vehicle)
        elif args.mission == "test-takeoff":
            success = test_takeoff(vehicle, args.altitude)
        elif args.mission == "test-camera":
            success = test_camera()
        elif args.mission == "test-detect":
            success = test_detection(args.model)
        elif args.mission == "test-motor":
            success = test_motor(vehicle, args.throttle)
        elif args.mission == "waypoint":
            success = mission_waypoint(vehicle, args.altitude)
        elif args.mission == "waypoint-detect":
            success = mission_waypoint_detect(vehicle, args.altitude, args.model)
        elif args.mission == "package-delivery":
            success = mission_package_delivery(vehicle, args.altitude, args.model)
        elif args.mission == "package-drop":
            success = mission_package_drop(vehicle, args.altitude, args.model)
        elif args.mission == "target-localize":
            success = mission_target_localize(vehicle, args.altitude)
        elif args.mission == "fix-mode":
            # Fix the mode after RTL
            if not vehicle:
                logging.error("Vehicle connection required for fix-mode")
                return 1
            success = set_mode(vehicle, "LOITER")
            if success:
                logging.info("Successfully changed vehicle mode to LOITER")
            else:
                logging.error("Failed to change vehicle mode")
        elif args.mission == "diagnostics":
            from drone.connection import get_vehicle_diagnostics
            diagnostics = get_vehicle_diagnostics(vehicle, timeout=10)
            if diagnostics:
                success = True
                logging.info("Diagnostics complete - see log for details")
            else:
                success = False
                logging.error("Failed to get diagnostics")
        elif args.mission == "reset-controller":
            from drone.connection import reset_flight_controller
            success = reset_flight_controller(vehicle)
            if success:
                logging.info("Reset command sent to flight controller")
            else:
                logging.error("Failed to send reset command")
        elif args.mission == "safety-check":
            from drone.navigation import run_preflight_checks
            checks_passed, failure_reason = run_preflight_checks(vehicle)
            success = checks_passed
            if not success:
                logging.error(f"Safety checks failed: {failure_reason}")
            else:
                logging.info("All safety checks passed!")

        elif args.mission == "orientation-check":
            from drone.navigation import verify_orientation
            success = verify_orientation(vehicle)
            if success:
                logging.info("Orientation is stable and suitable for takeoff")
            else:
                logging.warning("Orientation may be unstable - use caution")

        elif args.mission == "incremental-takeoff":
            from missions.test_missions import test_incremental_takeoff
            success = test_incremental_takeoff(vehicle, args.altitude, args.increment)

        elif args.mission == "position-hold-check" :
            from drone.navigation import verify_position_hold
            success = verify_position_hold(vehicle)
        elif args.mission == "check-altitude":
            from missions.test_missions import monitor_altitude_realtime
            success = monitor_altitude_realtime(vehicle, duration=0)  # 0 = indefinite
            if success:
                logging.info("Altitude monitoring completed")
            else:
                logging.error("Altitude monitoring failed")
        elif args.mission == "preflight-all":
            success = test_connection(vehicle)
            time.sleep(2)
            success = test_arm(vehicle)
            time.sleep(2)
            success = test_motor(vehicle, args.throttle)
            time.sleep(2)
            from drone.navigation import run_preflight_checks
            checks_passed, failure_reason = run_preflight_checks(vehicle)
            time.sleep(2)
            from drone.navigation import verify_orientation
            success = verify_orientation(vehicle)
            time.sleep(2)
            from drone.navigation import verify_position_hold
            success = verify_position_hold(vehicle)

        elif args.mission == "diamond-waypoints":
            from missions.waypoint import mission_diamond_precision
            success = mission_diamond_precision(vehicle, args.altitude)
            if success:
                logging.info("Diamond waypoint mission completed successfully")
            else:
                logging.error("Diamond waypoint mission failed")



        if success:
            logging.info(f"Mission '{args.mission}' completed successfully")
            return 0
        else:
            logging.error(f"Mission '{args.mission}' failed")
            return 1

    except KeyboardInterrupt:
        logging.warning("Mission aborted by user")
        return 130
    except Exception as e:
        logging.exception(f"Mission failed with error: {str(e)}")
        return 1
    finally:
        # Clean up resources
        if vehicle and mission_requires_vehicle:
            close_vehicle(vehicle)
        logging.info("Mission clean-up completed")

if __name__ == "__main__":
    sys.exit(main())
