#!/usr/bin/env python3
"""
main.py - IMPROVED VERSION WITH VIDEO RECORDING & BULLSEYE DETECTION
Drone Control System - Main Launcher
-----------------------------------
This script serves as the main entry point for all drone operations.
It parses command-line arguments to determine which mission to run.
"""

import argparse
import sys
import logging
import time
from typing import Dict, Callable, Any
from missions.waypoint_comp_area_bullseye import mission_competition_area_bullseye
from missions.waypoint_comp_area_gcp import mission_competition_area_gcp
from pymavlink import mavutil

# Import modules
from detection.gcp_yolo_detector import test_gcp_yolo_detection
from drone.connection import connect_vehicle, close_vehicle
from drone.navigation import set_mode, test_motors
from drone.servo import close_claw, idle_claw, open_claw, test_servo_simple
from missions.test_missions import (
    test_connection, test_arm, test_takeoff, test_camera,
    test_motor, test_incremental_takeoff, monitor_altitude_realtime
)
from missions.waypoint import (
    mission_diamond_precision_fixed, mission_waypoint
)
from detection.video_recorder import create_video_recorder
from detection.bullseye_detector import test_bullseye_detection
from missions.waypoint_bullseye import mission_waypoint_bullseye_detection
from missions.waypoint_gcp import mission_waypoint_gcp_detection
from missions.mission_parser import test_mission_parser

class MissionConfig:
    """Configuration class for mission parameters and shortcuts"""

    # Mission aliases mapping - maps shortcuts to primary mission names
    MISSION_ALIASES = {
        # Connection tests
        "test-connection": ["conn", "c", "1"],
        "preflight-all": ["pre", "p", "0"],
        "reset-controller": ["reset", "r", "2"],

        # Takeoff tests
        "incremental-takeoff": ["t-t", "inc-takeoff"],
        "diamond-waypoints": ["t-w", "diamond"],

        # Quick access shortcuts
        "test-arm": ["arm", "a"],
        "test-motor": ["motor", "m"],
        "test-camera": ["cam", "camera"],
        "test-servo": ["servo", "s"],
        "test-takeoff": ["takeoff", "to"],

        # NEW: Bullseye detection test
        "test-bullseye-video": ["bullseye", "bull", "b", "target"],
        "test-gcp-detection": ["gcp", "gcp-test", "g", "ground-control"],  # NEW: GCP detection

        # NEW: Video recording test
        "test-video-recording": ["record-test", "rec-test", "video-test"],

        # Diagnostics
        "diagnostics": ["diag", "d"],
        "check-altitude": ["alt", "altitude"],
        "safety-check": ["safety", "safe"],
        "orientation-check": ["orient", "orientation"],
        "position-hold-check": ["pos-hold", "position"],

        "test-mission-parser": ["parser", "test-parser", "mission-test", "mp"],

        "test-waypoint-bullseye": ["waypoint-bullseye", "wb", "bullseye-waypoint"],

        # GCP Detection missions
        "test-gcp-yolo": ["gcp", "gcp-yolo", "test-gcp", "gcp-test"],
        "test-waypoint-gcp": ["waypoint-gcp", "wgcp", "gcp-waypoint"],

        "test-comp-area-gcp": ["comp-area", "comp-gcp", "area-gcp", "competition-area"],
        "test-comp-area-bullseye": ["comp-bullseye", "comp-bull", "area-bullseye", "area-bull", "competition-bullseye"],

        "test-claw": ["claw"],
        "close-claw": ["close"],
    }

    # Reverse mapping for quick lookup
    ALIAS_TO_MISSION = {}
    for mission, aliases in MISSION_ALIASES.items():
        ALIAS_TO_MISSION[mission] = mission  # Add the primary name
        for alias in aliases:
            ALIAS_TO_MISSION[alias] = mission

class DroneController:
    """Main drone controller class"""

    def __init__(self):
        self.vehicle = None
        self.config = MissionConfig()
        self.mission_handlers = self._setup_mission_handlers()
        self.video_recorder = None

    def _setup_mission_handlers(self) -> Dict[str, Callable]:
        """Setup mission handler mapping"""
        return {
            # Connection and diagnostics
            "test-connection": self._handle_test_connection,
            "preflight-all": self._handle_preflight_all,
            "reset-controller": self._handle_reset_controller,

            # Basic tests
            "test-arm": self._handle_test_arm,
            "test-motor": self._handle_test_motor,
            "test-camera": self._handle_test_camera,
            "test-servo": self._handle_test_servo,

            # NEW: Video recording test
            "test-video-recording": self._handle_test_video_recording,

            # Takeoff tests
            "test-takeoff": self._handle_test_takeoff,
            "incremental-takeoff": self._handle_incremental_takeoff,
            "diamond-waypoints": self._handle_diamond_waypoints,

            # Navigation missions
            "waypoint": self._handle_waypoint,

            # System controls
            "fix-mode": self._handle_fix_mode,

            # Safety checks
            "diagnostics": self._handle_diagnostics,
            "safety-check": self._handle_safety_check,
            "orientation-check": self._handle_orientation_check,
            "position-hold-check": self._handle_position_hold_check,
            "check-altitude": self._handle_check_altitude,

            # Video Tests
            "test-bullseye-video": self._handle_test_bullseye_video,
            "test-gcp-yolo": self._handle_test_gcp_yolo,

            "test-mission-parser": self._handle_test_mission_parser,

            "test-waypoint-bullseye": self._handle_waypoint_bullseye,
            "test-waypoint-gcp": self._handle_waypoint_gcp,

            "test-claw": self._test_claw,
            "close-claw": self._close_claw,

            "test-comp-area-gcp": self._handle_comp_area_gcp,
            "test-comp-area-bullseye": self._handle_comp_area_bullseye,
        }



    def setup_logging(self):
        """Configure logging"""
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

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(
            description="Drone Mission Control System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_help_epilog()
        )

        # Get all possible mission choices (primary names + aliases)
        all_choices = list(self.config.ALIAS_TO_MISSION.keys())

        parser.add_argument(
            "mission",
            choices=all_choices,
            help="Mission to execute (use -h to see shortcuts)"
        )

        # Mission parameters
        parser.add_argument("--altitude", type=float, default=3.0,
                          help="Target altitude in meters")
        parser.add_argument("--connection", type=str, default="tcp:127.0.0.1:5761",
                          help="Connection string for the vehicle")
        parser.add_argument("--model", type=str, default="models/latest.pt",
                          help="Path to the detection model")
        parser.add_argument("--duration", type=float, default=1.0,
                          help="Duration for motor testing")
        parser.add_argument("--throttle", type=float, default=15.0,
                          help="Throttle percentage for motor testing (0-100)")
        parser.add_argument("--increment", type=float, default=1.0,
                          help="Height increment in meters for incremental takeoff")
        parser.add_argument("--loops", type=int, default=1,
                          help="Number of times to repeat the mission")

        # NEW: Video recording options
        parser.add_argument("--record", action="store_true",
                          help="Record video during mission execution")
        parser.add_argument("--record-fps", type=float, default=30.0,
                          help="Recording frame rate (default: 30 FPS)")
        parser.add_argument("--record-dir", type=str, default="recordings",
                          help="Directory to save recordings (default: recordings)")

        # NEW: Bullseye detection options
        parser.add_argument("--source", type=str, default="0",
                          help="Camera ID, video file, or image file for bullseye detection")
        parser.add_argument("--display", action="store_true", default=True,
                          help="Display detection results (default: True)")
        parser.add_argument("--no-display", dest="display", action="store_false",
                          help="Disable display of detection results")
        parser.add_argument("--save-results", action="store_true", default=True,
                          help="Save detection results (default: True)")
        parser.add_argument("--no-save", dest="save_results", action="store_false",
                          help="Disable saving of detection results")
        parser.add_argument("--video-delay", type=float, default=0.1,
                          help="Delay between video frames in seconds (default: 0.1)")

        parser.add_argument("--confidence", type=float, default=0.5,
                          help="Detection confidence threshold (default: 0.5)")


        # GCP-specific options
        parser.add_argument("--gcp-model", type=str, default="models/best-gcp.pt",
                          help="Path to GCP YOLO model (default: models/best-gcp.pt)")
        parser.add_argument("--gcp-confidence", type=float, default=0.5,
                          help="Confidence threshold for GCP detection (default: 0.5)")

        parser.add_argument("--mission-file", type=str,
                          help="Path to Mission Planner .mission file")

        # NEW: Action parameters for waypoint bullseye mission
        action_group = parser.add_mutually_exclusive_group()
        action_group.add_argument("--land", action="store_const", dest="action", const="land",
                                help="Land on bullseye when detected (default behavior)")
        action_group.add_argument("--drop", action="store_const", dest="action", const="drop",
                                help="Drop payload at altitude when bullseye detected, then RTL")
        action_group.add_argument("--deliver", action="store_const", dest="action", const="deliver",
                                help="Descend to 1-2m above bullseye, drop payload, then RTL")

        # Set default action if none specified
        parser.set_defaults(action="land")


        return parser

    def _get_help_epilog(self) -> str:
        """Generate help text showing mission shortcuts"""
        lines = ["\nMission Shortcuts:"]
        lines.append("=" * 50)

        for mission, aliases in self.config.MISSION_ALIASES.items():
            if aliases:  # Only show if there are aliases
                aliases_str = ", ".join(aliases)
                lines.append(f"  {mission:<20} → {aliases_str}")

        lines.append("\nExamples:")
        lines.append("  python main.py c              # test-connection")
        lines.append("  python main.py pre --altitude 5  # preflight-all at 5m")
        lines.append("  python main.py t-t --increment 0.5  # incremental takeoff")
        lines.append("  python main.py diamond --loops 3    # 3 diamond loops")
        lines.append("  python main.py diamond --record     # record video during mission")
        lines.append("  python main.py bullseye --source video.mp4  # detect bullseyes in video")
        lines.append("  python main.py bull --source image.jpg --no-display  # process image without display")
        lines.append("  python main.py record-test --duration 15  # test video recording for 15 seconds")
        lines.append("  python main.py rec-test --source 1 --duration 30  # test camera 1 for 30 seconds")

        lines.append("  python main.py wb --altitude 5 --loops 2    # waypoint bullseye mission")
        lines.append("  python main.py test-waypoint-bullseye --model best.pt --confidence 0.6")

        lines.append("  python main.py gcp --source video.mp4  # detect GCP markers in video")
        lines.append("  python main.py test-gcp-yolo --source 0 --gcp-confidence 0.6  # test GCP detection")
        lines.append("  python main.py wgcp --altitude 8 --loops 1    # waypoint GCP mission")
        lines.append("  python main.py test-waypoint-gcp --gcp-model best-gcp.pt --confidence 0.7")

        lines.append("  python main.py test-parser --source test.mission  # test mission parser")
        lines.append("  python main.py diamond --mission-file waypoints.mission --altitude 8  # use mission file with altitude override")


        lines.append("  python main.py comp-area --altitude 8 --gcp-confidence 0.6  # competition area GCP mission")
        lines.append("  python main.py test-comp-area-gcp --gcp-model best-gcp.pt --altitude 10")

        return "\n".join(lines)

    def connect_if_needed(self, mission: str, connection_string: str) -> bool:
        """Connect to vehicle if the mission requires it"""
        # Resolve mission alias to primary name FIRST
        primary_mission = self.config.ALIAS_TO_MISSION.get(mission, mission)

        missions_without_vehicle = {
            "test-mission-parser",
            "test-camera",
            "test-bullseye-video",
            "test-video-recording",
            "test-gcp-detection",
            "test-gcp-yolo",
        }

        if primary_mission in missions_without_vehicle:
            logging.info(f"Mission '{primary_mission}' does not require vehicle connection")
            return True

        logging.info(f"Connecting to vehicle at {connection_string}")
        self.vehicle = connect_vehicle(connection_string)

        if not self.vehicle:
            logging.error("Failed to connect to vehicle")
            return False

        return True

    def start_recording_if_requested(self, args, mission_name: str) -> bool:
        """Start video recording if requested"""
        if not args.record:
            return True

        logging.info("Starting video recording...")
        self.video_recorder = create_video_recorder(
            output_dir=args.record_dir,
            fps=args.record_fps
        )

        success = self.video_recorder.start_recording(
            camera_id=0,
            mission_name=mission_name
        )

        if success:
            logging.info(f"Recording started for mission: {mission_name}")
        else:
            logging.error("Failed to start video recording")
            self.video_recorder = None

        return success

    def stop_recording_if_active(self):
        """Stop video recording if active"""
        if self.video_recorder and self.video_recorder.is_recording():
            logging.info("Stopping video recording...")
            success = self.video_recorder.stop_recording()
            if success:
                logging.info("Video recording stopped successfully")
            else:
                logging.error("Error stopping video recording")
            self.video_recorder = None

    def execute_mission(self, mission: str, args: argparse.Namespace) -> bool:
        """Execute the specified mission"""
        # Resolve mission alias to primary name
        primary_mission = self.config.ALIAS_TO_MISSION.get(mission, mission)

        # Get handler for the mission
        handler = self.mission_handlers.get(primary_mission)

        if not handler:
            logging.error(f"Unknown mission: {mission}")
            return False

        logging.info(f"Executing mission: {primary_mission} (alias: {mission})")

        try:
            # Start recording if requested (except for standalone test missions)
            standalone_missions = {"test-camera", "test-bullseye-video", "test-gcp-detection", "test-video-recording"}
            if primary_mission not in standalone_missions:
                if not self.start_recording_if_requested(args, primary_mission):
                    logging.warning("Continuing mission without recording")

            # Execute the mission
            result = handler(args)

            return result

        except Exception as e:
            logging.exception(f"Mission '{primary_mission}' failed with error: {str(e)}")
            return False
        finally:
            # Always stop recording when mission ends
            self.stop_recording_if_active()

    # Mission handler methods
    def _handle_test_connection(self, args) -> bool:
        return test_connection(self.vehicle)

    def _handle_test_arm(self, args) -> bool:
        return test_arm(self.vehicle)

    def _handle_test_takeoff(self, args) -> bool:
        return test_takeoff(self.vehicle, args.altitude)

    def _handle_test_camera(self, args) -> bool:
        return test_camera()

    def _handle_test_motor(self, args) -> bool:
        return test_motor(self.vehicle, args.throttle, args.duration)

    def _handle_test_servo(self, args) -> bool:
        return test_servo_simple(self.vehicle)

    def _test_claw(self, args) -> bool:
        open_claw(self.vehicle)
        close_claw(self.vehicle)
        idle_claw(self.vehicle)

    def _close_claw(self, args) -> bool:
        close_claw(self.vehicle)
        idle_claw(self.vehicle)


    def _handle_test_bullseye_video(self, args) -> bool:
        """Handle bullseye detection test"""
        # Convert source to appropriate type
        source = args.source
        try:
            # Try to convert to int (camera ID)
            source = int(source)
        except ValueError:
            # Keep as string (file path)
            pass

        return test_bullseye_detection(
            source=source,
            display=args.display,
            save_results=args.save_results,
            duration=args.duration if isinstance(source, int) else 0,
            video_delay=args.video_delay,
            model_path=args.model,
        )

    def _handle_test_video_recording(self, args) -> bool:
        """Handle video recording test (no vehicle required)"""
        from detection.video_recorder import test_video_recording

        # Convert source to camera ID if it's a digit
        camera_id = 0
        try:
            camera_id = int(args.source)
        except ValueError:
            logging.warning(f"Invalid camera ID '{args.source}', using camera 0")

        return test_video_recording(
            camera_id=camera_id,
            duration=args.duration if args.duration > 1 else 10,  # Default 10 seconds
            output_dir=args.record_dir
        )

    def _handle_incremental_takeoff(self, args) -> bool:
        return test_incremental_takeoff(self.vehicle, args.altitude, args.increment)

    def _handle_diamond_waypoints(self, args) -> bool:
        mission_file = getattr(args, 'mission_file', None) or getattr(args, 'source', None)
        return mission_diamond_precision_fixed(self.vehicle, args.altitude, args.loops, mission_file)

    def _handle_waypoint(self, args) -> bool:
        return mission_waypoint(self.vehicle, args.altitude)

    def _handle_fix_mode(self, args) -> bool:
        if not self.vehicle:
            logging.error("Vehicle connection required for fix-mode")
            return False

        success = set_mode(self.vehicle, "LOITER")
        if success:
            logging.info("Successfully changed vehicle mode to LOITER")
        else:
            logging.error("Failed to change vehicle mode")
        return success

    def _handle_diagnostics(self, args) -> bool:
        from drone.connection import get_vehicle_diagnostics
        diagnostics = get_vehicle_diagnostics(self.vehicle, timeout=10)
        if diagnostics:
            logging.info("Diagnostics complete - see log for details")
            return True
        else:
            logging.error("Failed to get diagnostics")
            return False

    def _handle_reset_controller(self, args) -> bool:
        from drone.connection import reset_flight_controller
        success = reset_flight_controller(self.vehicle)
        if success:
            logging.info("Reset command sent to flight controller")
        else:
            logging.error("Failed to send reset command")
        return success

    def _handle_safety_check(self, args) -> bool:
        from drone.navigation import run_preflight_checks
        checks_passed, failure_reason = run_preflight_checks(self.vehicle)
        if not checks_passed:
            logging.error(f"Safety checks failed: {failure_reason}")
        else:
            logging.info("All safety checks passed!")
        return checks_passed

    def _handle_orientation_check(self, args) -> bool:
        from drone.navigation import verify_orientation
        success = verify_orientation(self.vehicle)
        if success:
            logging.info("Orientation is stable and suitable for takeoff")
        else:
            logging.warning("Orientation may be unstable - use caution")
        return success

    def _handle_position_hold_check(self, args) -> bool:
        from drone.navigation import verify_position_hold
        return verify_position_hold(self.vehicle)

    def _handle_check_altitude(self, args) -> bool:
        success = monitor_altitude_realtime(self.vehicle, duration=0)
        if success:
            logging.info("Altitude monitoring completed")
        else:
            logging.error("Altitude monitoring failed")
        return success

    def _handle_waypoint_bullseye(self, args) -> bool:
        """Handle waypoint bullseye detection and landing mission"""
        return mission_waypoint_bullseye_detection(
            vehicle=self.vehicle,
            altitude=args.altitude,
            model_path=args.model,
            confidence=getattr(args, 'confidence', 0.5),
            loops=args.loops,
            land_on_detection=True,
            video_recorder=self.video_recorder,  # Pass the shared video recorder
            action=args.action
        )

    def _handle_test_gcp_yolo(self, args) -> bool:
        """Handle GCP YOLO detection test"""
        # Convert source to appropriate type
        source = args.source
        try:
            # Try to convert to int (camera ID)
            source = int(source)
        except ValueError:
            # Keep as string (file path)
            pass

        return test_gcp_yolo_detection(
            source=source,
            display=args.display,
            save_results=args.save_results,
            duration=args.duration if isinstance(source, int) else 0,
            video_delay=args.video_delay,
            model_path=args.gcp_model,
            confidence=args.gcp_confidence,
            imgsz=getattr(args, 'imgsz', 160)
        )

    def _handle_waypoint_gcp(self, args) -> bool:
        """Handle waypoint GCP detection and collection mission"""
        return mission_waypoint_gcp_detection(
            vehicle=self.vehicle,
            altitude=args.altitude,
            model_path=args.gcp_model,
            confidence=args.gcp_confidence,
            loops=args.loops,
            video_recorder=self.video_recorder  # Pass the shared video recorder
        )

    def _handle_test_mission_parser(self, args) -> bool:
        """Handle mission parser test"""
        mission_file = getattr(args, 'mission_file', None) or getattr(args, 'source', None)

        if not mission_file:
            logging.error("Mission file required for parser test. Use --mission-file or --source parameter")
            return False

        return test_mission_parser(mission_file)
    def _handle_comp_area_gcp(self, args) -> bool:
        """Handle competition area GCP detection mission"""
        return mission_competition_area_gcp(
            vehicle=self.vehicle,
            altitude=args.altitude,
            model_path=args.gcp_model,
            confidence=args.gcp_confidence,
            video_recorder=self.video_recorder
        )

    def _handle_comp_area_bullseye(self, args) -> bool:
        """Handle competition area bullseye detection mission"""
        return mission_competition_area_bullseye(
            vehicle=self.vehicle,
            altitude=args.altitude,
            model_path=args.model,
            confidence=args.confidence,
            video_recorder=self.video_recorder,
            action=args.action
        )


    def _handle_preflight_all(self, args) -> bool:
        """Run comprehensive preflight checks"""
        logging.info("Running comprehensive preflight checks...")

        checks = [
            ("Connection Test", lambda: test_connection(self.vehicle)),
            ("Arm Test", lambda: test_arm(self.vehicle)),
            ("Motor Test", lambda: test_motor(self.vehicle, args.throttle)),
            ("Safety Checks", self._handle_safety_check),
            ("Orientation Check", self._handle_orientation_check),
            ("Position Hold Check", self._handle_position_hold_check),
        ]

        results = []
        for check_name, check_func in checks:
            logging.info(f"Running {check_name}...")
            try:
                result = check_func(args) if check_name in ["Safety Checks", "Orientation Check", "Position Hold Check"] else check_func()
                results.append((check_name, result))
                logging.info(f"{check_name}: {'PASSED' if result else 'FAILED'}")
                time.sleep(2)  # Brief pause between checks
            except Exception as e:
                logging.error(f"{check_name} failed with exception: {str(e)}")
                results.append((check_name, False))

        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)

        logging.info(f"\nPreflight Summary: {passed}/{total} checks passed")
        for check_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            logging.info(f"  {check_name}: {status}")

        return passed == total

    def cleanup(self):
        """Clean up resources"""
        # Stop recording if active
        self.stop_recording_if_active()

        # Close vehicle connection
        if self.vehicle:
            close_vehicle(self.vehicle)
            logging.info("Mission clean-up completed")

    def run(self) -> int:
        """Main execution method"""
        try:
            self.setup_logging()
            parser = self.create_parser()
            args = parser.parse_args()

            # Connect to vehicle if needed
            if not self.connect_if_needed(args.mission, args.connection):
                return 1

            # Execute mission
            success = self.execute_mission(args.mission, args)

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
            logging.exception(f"Unexpected error: {str(e)}")
            return 1
        finally:
            self.cleanup()

def main():
    """Entry point"""
    controller = DroneController()
    return controller.run()

if __name__ == "__main__":
    sys.exit(main())
