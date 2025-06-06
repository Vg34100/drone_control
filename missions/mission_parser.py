# missions/mission_parser.py - NEW MODULE
"""
Mission Planner File Parser
---------------------------
Parse Mission Planner .mission files and extract waypoint information.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional

class MissionWaypoint:
    """Class to represent a single waypoint from a mission file"""

    def __init__(self, seq: int, command: int, param1: float, param2: float,
                 param3: float, param4: float, x: float, y: float, z: float,
                 autocontinue: int = 1, frame: int = 3):
        self.seq = seq
        self.command = command
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
        self.x = x  # longitude (-119.xxx)
        self.y = y  # latitude (35.xxx)
        self.z = z  # altitude
        self.autocontinue = autocontinue
        self.frame = frame

    @property
    def latitude(self) -> float:
        """Get latitude (y coordinate) - should be 35.xxx"""
        return self.y

    @property
    def longitude(self) -> float:
        """Get longitude (x coordinate) - should be -119.xxx"""
        return self.x

    @property
    def altitude(self) -> float:
        """Get altitude (z coordinate)"""
        return self.z

    def __str__(self) -> str:
        return f"WP{self.seq}: Lat={self.latitude:.7f}, Lon={self.longitude:.7f}, Alt={self.altitude:.1f}m"

class MissionParser:
    """Parser for Mission Planner .mission files"""

    def __init__(self):
        self.waypoints = []
        self.home_location = None
        self.version = None

    def parse_mission_file(self, file_path: str) -> bool:
        """
        Parse a Mission Planner mission file.

        Args:
            file_path: Path to the .mission file

        Returns:
            True if parsing successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"Mission file not found: {file_path}")
                return False

            logging.info(f"Parsing mission file: {file_path}")

            with open(file_path, 'r') as file:
                lines = file.readlines()

            if not lines:
                logging.error("Mission file is empty")
                return False

            # Parse header (first line should be version info)
            first_line = lines[0].strip()
            if first_line.startswith("QGC WPL"):
                self.version = first_line
                logging.info(f"Mission file version: {self.version}")
                lines = lines[1:]  # Skip version line
            else:
                logging.warning("No QGC WPL header found, assuming old format")

            # Clear existing waypoints
            self.waypoints = []
            self.home_location = None

            # Parse each waypoint line
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    waypoint = self._parse_waypoint_line(line)
                    if waypoint:
                        # Check if this is home location (sequence 0)
                        if waypoint.seq == 0:
                            self.home_location = waypoint
                            logging.info(f"Home location: Lat={waypoint.latitude:.7f}, Lon={waypoint.longitude:.7f}, Alt={waypoint.altitude:.1f}m")
                        else:
                            self.waypoints.append(waypoint)

                except Exception as e:
                    logging.warning(f"Error parsing line {line_num}: {line} - {str(e)}")
                    continue

            logging.info(f"Successfully parsed {len(self.waypoints)} waypoints from mission file")
            return True

        except Exception as e:
            logging.error(f"Error parsing mission file: {str(e)}")
            return False

    def _parse_waypoint_line(self, line: str) -> Optional[MissionWaypoint]:
        """
        Parse a single waypoint line from the mission file.

        Mission file format:
        seq current wp coord_frame command param1 param2 param3 param4 lat lon alt autocontinue
        """
        try:
            parts = line.split('\t')
            if len(parts) < 11:
                logging.warning(f"Invalid waypoint line format (expected 12 fields): {line}")
                return None

            seq = int(parts[0])
            current = int(parts[1])  # Not used
            frame = int(parts[2])
            command = int(parts[3])
            param1 = float(parts[4])
            param2 = float(parts[5])
            param3 = float(parts[6])
            param4 = float(parts[7])
            x = float(parts[8])  # latitude in your format
            y = float(parts[9])  # longitude in your format
            z = float(parts[10])  # altitude
            autocontinue = int(parts[11]) if len(parts) > 11 else 1

            # NOTE: In your mission file format, parts[8] is lat and parts[9] is lon
            # So we need to swap: longitude=y, latitude=x
            return MissionWaypoint(seq, command, param1, param2, param3, param4,
                                 y, x, z, autocontinue, frame)  # swapped x and y

        except (ValueError, IndexError) as e:
            logging.warning(f"Error parsing waypoint line: {line} - {str(e)}")
            return None

    def get_navigation_waypoints(self) -> List[Tuple[float, float, float]]:
        """
        Get only the navigation waypoints (excluding non-navigation commands).

        Returns:
            List of (latitude, longitude, altitude) tuples
        """
        nav_waypoints = []

        # Common navigation commands in Mission Planner
        nav_commands = {
            16,   # MAV_CMD_NAV_WAYPOINT
            17,   # MAV_CMD_NAV_LOITER_UNLIM
            18,   # MAV_CMD_NAV_LOITER_TURNS
            19,   # MAV_CMD_NAV_LOITER_TIME
            20,   # MAV_CMD_NAV_RETURN_TO_LAUNCH
            21,   # MAV_CMD_NAV_LAND
            22,   # MAV_CMD_NAV_TAKEOFF
            82,   # MAV_CMD_NAV_SPLINE_WAYPOINT
        }

        for wp in self.waypoints:
            if wp.command in nav_commands:
                nav_waypoints.append((wp.latitude, wp.longitude, wp.altitude))

        return nav_waypoints

    def get_simple_waypoints(self) -> List[Tuple[float, float]]:
        """
        Get simple waypoints as (latitude, longitude) pairs for existing diamond mission.

        Returns:
            List of (latitude, longitude) tuples
        """
        return [(wp.latitude, wp.longitude) for wp in self.waypoints
                if wp.command in {16, 82}]  # Only basic waypoints and spline waypoints

    def override_altitudes(self, new_altitude: float) -> None:
        """
        Override all waypoint altitudes with a new altitude.

        Args:
            new_altitude: New altitude in meters
        """
        for wp in self.waypoints:
            wp.z = new_altitude

        logging.info(f"Overrode all waypoint altitudes to {new_altitude}m")

    def print_mission_summary(self) -> None:
        """Print a summary of the parsed mission"""
        print("\n" + "="*60)
        print("MISSION FILE SUMMARY")
        print("="*60)

        if self.version:
            print(f"Version: {self.version}")

        if self.home_location:
            print(f"Home: Lat={self.home_location.latitude:.7f}, Lon={self.home_location.longitude:.7f}, Alt={self.home_location.altitude:.1f}m")

        print(f"Total waypoints: {len(self.waypoints)}")

        if self.waypoints:
            print(f"\nWaypoint details:")
            for i, wp in enumerate(self.waypoints, 1):
                cmd_name = self._get_command_name(wp.command)
                print(f"  {i:2d}. {wp} - {cmd_name}")

        nav_waypoints = self.get_navigation_waypoints()
        print(f"\nNavigation waypoints: {len(nav_waypoints)}")

        print("="*60)

    def _get_command_name(self, command: int) -> str:
        """Get human-readable command name"""
        command_names = {
            16: "WAYPOINT",
            17: "LOITER_UNLIM",
            18: "LOITER_TURNS",
            19: "LOITER_TIME",
            20: "RTL",
            21: "LAND",
            22: "TAKEOFF",
            82: "SPLINE_WAYPOINT",
            112: "CONDITION_DELAY",
            113: "CONDITION_CHANGE_ALT",
            114: "CONDITION_DISTANCE",
            115: "CONDITION_YAW",
            177: "DO_JUMP",
            183: "DO_CHANGE_SPEED",
            206: "DO_SET_HOME"
        }
        return command_names.get(command, f"CMD_{command}")

def test_mission_parser(file_path: str) -> bool:
    """
    Test function to parse and display mission file contents.

    Args:
        file_path: Path to the mission file to test

    Returns:
        True if test successful, False otherwise
    """
    try:
        logging.info(f"=== MISSION PARSER TEST ===")
        logging.info(f"Testing mission file: {file_path}")

        # Create parser and parse file
        parser = MissionParser()
        success = parser.parse_mission_file(file_path)

        if not success:
            logging.error("Failed to parse mission file")
            return False

        # Print detailed summary
        parser.print_mission_summary()

        # Test different waypoint extraction methods
        nav_waypoints = parser.get_navigation_waypoints()
        simple_waypoints = parser.get_simple_waypoints()

        # Debug: Show first waypoint details
        if parser.waypoints:
            first_wp = parser.waypoints[0]
            print(f"\nDEBUG - First waypoint raw data:")
            print(f"  x (longitude): {first_wp.x}")
            print(f"  y (latitude): {first_wp.y}")
            print(f"  z (altitude): {first_wp.z}")
            print(f"  latitude property: {first_wp.latitude}")
            print(f"  longitude property: {first_wp.longitude}")

        print(f"\nExtracted navigation waypoints ({len(nav_waypoints)}):")
        for i, (lat, lon, alt) in enumerate(nav_waypoints, 1):
            print(f"  NAV {i:2d}: Lat={lat:.7f}, Lon={lon:.7f}, Alt={alt:.1f}m")

        print(f"\nExtracted simple waypoints ({len(simple_waypoints)}):")
        for i, (lat, lon) in enumerate(simple_waypoints, 1):
            print(f"  WP  {i:2d}: Lat={lat:.7f}, Lon={lon:.7f}")

        # Test altitude override
        if nav_waypoints:
            print(f"\nTesting altitude override to 10m...")
            parser.override_altitudes(10.0)
            updated_waypoints = parser.get_navigation_waypoints()
            print(f"Updated waypoints:")
            for i, (lat, lon, alt) in enumerate(updated_waypoints, 1):
                print(f"  UPD {i:2d}: Lat={lat:.7f}, Lon={lon:.7f}, Alt={alt:.1f}m")

        logging.info("Mission parser test completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error during mission parser test: {str(e)}")
        return False

def create_mission_parser():
    """
    Factory function to create a mission parser.

    Returns:
        MissionParser instance
    """
    return MissionParser()
