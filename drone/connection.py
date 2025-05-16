"""
Drone Connection Module
----------------------
Functions for connecting to and managing the drone vehicle using pymavlink.
"""

import time
import logging
import math
from pymavlink import mavutil

def connect_vehicle(connection_string):
    """
    Connect to the drone vehicle using pymavlink.

    Args:
        connection_string: The connection string (e.g., 'tcp:127.0.0.1:5761')

    Returns:
        The connected mavlink object or None if connection failed
    """
    try:
        logging.info(f"Connecting to vehicle on: {connection_string}")

        # Parse the connection string
        if connection_string.startswith('tcp:'):
            # TCP connection (e.g., 'tcp:127.0.0.1:5761')
            vehicle = mavutil.mavlink_connection(connection_string)
        elif connection_string.startswith('udp:'):
            # UDP connection (e.g., 'udp:127.0.0.1:14550')
            vehicle = mavutil.mavlink_connection(connection_string)
        elif ',' in connection_string:
            # Serial connection with baud rate (e.g., '/dev/ttyUSB0,57600')
            parts = connection_string.split(',')
            if len(parts) == 2:
                port, baud = parts
                vehicle = mavutil.mavlink_connection(port, baud=int(baud))
            else:
                logging.error(f"Invalid serial connection string: {connection_string}")
                return None
        else:
            # Assume it's a serial port with default baud rate
            vehicle = mavutil.mavlink_connection(connection_string, baud=57600)

        # Wait for the heartbeat to ensure connection is established
        logging.info("Waiting for heartbeat...")
        vehicle.wait_heartbeat()

        logging.info(f"Connected to vehicle (system: {vehicle.target_system}, component: {vehicle.target_component})")
        return vehicle
    except Exception as e:
        logging.error(f"Error connecting to vehicle: {str(e)}")
        return None

def close_vehicle(vehicle):
    """
    Safely close the connection to the vehicle.

    Args:
        vehicle: The connected mavlink object
    """
    if vehicle:
        try:
            vehicle.close()
            logging.info("Vehicle connection closed")
        except Exception as e:
            logging.error(f"Error closing vehicle connection: {str(e)}")

def get_vehicle_state(vehicle):
    """
    Get the current state of the vehicle.

    Args:
        vehicle: The connected mavlink object

    Returns:
        Dictionary containing vehicle state information
    """
    if not vehicle:
        return None

    try:
        # Request current parameters and system status
        vehicle.mav.param_request_list_send(
            vehicle.target_system, vehicle.target_component
        )

        vehicle.mav.request_data_stream_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            1,  # Rate in Hz
            1   # Start/stop (1=start, 0=stop)
        )

        # Initialize state dictionary
        state = {
            "mode": None,
            "armed": None,
            "system_status": None,
            "gps_fix_type": None,
            "altitude": None,
            "location": {"lat": None, "lon": None, "alt": None},
            "attitude": {"roll": None, "pitch": None, "yaw": None},
            "velocity": {"vx": None, "vy": None, "vz": None},
            "battery": {"voltage": None, "current": None, "remaining": None}
        }

        # Wait for and process messages to populate state data
        start_time = time.time()
        timeout = 3  # seconds

        while time.time() - start_time < timeout:
            msg = vehicle.recv_match(blocking=False)
            if not msg:
                time.sleep(0.1)
                continue

            msg_type = msg.get_type()

            if msg_type == "HEARTBEAT":
                state["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                state["mode"] = mavutil.mode_string_v10(msg)
                state["system_status"] = msg.system_status
            elif msg_type == "GLOBAL_POSITION_INT":
                state["location"]["lat"] = msg.lat / 1e7
                state["location"]["lon"] = msg.lon / 1e7
                state["location"]["alt"] = msg.alt / 1000.0  # Convert mm to m
                state["altitude"] = msg.relative_alt / 1000.0  # Convert mm to m
            elif msg_type == "ATTITUDE":
                state["attitude"]["roll"] = math.degrees(msg.roll)
                state["attitude"]["pitch"] = math.degrees(msg.pitch)
                state["attitude"]["yaw"] = math.degrees(msg.yaw)
            elif msg_type == "GPS_RAW_INT":
                state["gps_fix_type"] = msg.fix_type
            elif msg_type == "VFR_HUD":
                state["groundspeed"] = msg.groundspeed
                state["airspeed"] = msg.airspeed
                state["heading"] = msg.heading
            elif msg_type == "SYS_STATUS":
                state["battery"]["voltage"] = msg.voltage_battery / 1000.0  # Convert mV to V
                state["battery"]["current"] = msg.current_battery / 100.0  # Convert 10*mA to A
                state["battery"]["remaining"] = msg.battery_remaining

        return state
    except Exception as e:
        logging.error(f"Error getting vehicle state: {str(e)}")
        return None

def print_vehicle_state(vehicle):
    """
    Print the current state of the vehicle to the console.

    Args:
        vehicle: The connected mavlink object
    """
    state = get_vehicle_state(vehicle)
    if state:
        logging.info("===== Vehicle State =====")

        # Format mode and armed status
        logging.info(f"Mode: {state['mode']}")
        logging.info(f"Armed: {state['armed']}")

        # Format location
        lat = state['location']['lat']
        lon = state['location']['lon']
        alt = state['location']['alt']
        rel_alt = state['altitude']
        if lat is not None and lon is not None:
            logging.info(f"Location: Lat={lat}, Lon={lon}, Alt={alt}m (Relative Alt={rel_alt}m)")

        # Format attitude
        roll = state['attitude']['roll']
        pitch = state['attitude']['pitch']
        yaw = state['attitude']['yaw']
        if roll is not None:
            logging.info(f"Attitude: Roll={roll}°, Pitch={pitch}°, Yaw={yaw}°")

        # Format battery
        voltage = state['battery']['voltage']
        current = state['battery']['current']
        remaining = state['battery']['remaining']
        if voltage is not None:
            logging.info(f"Battery: Voltage={voltage}V, Current={current}A, Remaining={remaining}%")

        # Format GPS
        logging.info(f"GPS Fix Type: {state['gps_fix_type']}")

        # Format velocity
        if 'groundspeed' in state:
            logging.info(f"Groundspeed: {state['groundspeed']} m/s")
            logging.info(f"Airspeed: {state['airspeed']} m/s")
            logging.info(f"Heading: {state['heading']}°")

        logging.info("========================")
    else:
        logging.warning("Could not retrieve vehicle state")

def request_message_interval(vehicle, message_id, frequency_hz):
    """
    Request a specific mavlink message at a given frequency.

    Args:
        vehicle: The connected mavlink object
        message_id: The MAVLink message ID to request
        frequency_hz: The frequency in Hz to request (0 means stop)

    Returns:
        True if the request was sent, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Calculate message interval in microseconds
        if frequency_hz == 0:
            interval = 0  # 0 means stop
        else:
            interval = int(1000000 / frequency_hz)

        # Request message interval
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,                  # Confirmation
            message_id,         # Param 1: Message ID
            interval,           # Param 2: Interval in microseconds
            0, 0, 0, 0, 0       # Params 3-7 (not used)
        )

        return True
    except Exception as e:
        logging.error(f"Error setting message interval: {str(e)}")
        return False

def wait_for_message(vehicle, message_type, timeout=5):
    """
    Wait for a specific MAVLink message type.

    Args:
        vehicle: The connected mavlink object
        message_type: The MAVLink message type to wait for
        timeout: Maximum time to wait in seconds

    Returns:
        The received message or None if timeout
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return None

    try:
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = vehicle.recv_match(type=message_type, blocking=True, timeout=0.5)
            if msg:
                return msg

        logging.warning(f"Timeout waiting for {message_type} message")
        return None
    except Exception as e:
        logging.error(f"Error waiting for message: {str(e)}")
        return None

# --- drone/connection.py ---
def get_vehicle_diagnostics(vehicle, timeout=10):
    """
    Get comprehensive diagnostics for the vehicle.

    Args:
        vehicle: The connected mavlink object
        timeout: Maximum time to collect diagnostics in seconds

    Returns:
        Dictionary containing diagnostic information
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return None

    try:
        # Initialize diagnostics dictionary
        diagnostics = {
            "connection": {
                "target_system": getattr(vehicle, 'target_system', 'Unknown'),
                "target_component": getattr(vehicle, 'target_component', 'Unknown'),
                "connection_string": getattr(vehicle, 'address', 'Unknown'),
            },
            "heartbeat_received": False,
            "status_text_messages": [],
            "pre_arm_status": [],
            "gps_status": None,
            "mode": None,
            "armed": None,
            "params_received": False,
            "firmware_version": None
        }

        # Request parameters and system status
        vehicle.mav.param_request_list_send(
            getattr(vehicle, 'target_system', 1),
            getattr(vehicle, 'target_component', 0)
        )

        # Request data streams
        vehicle.mav.request_data_stream_send(
            getattr(vehicle, 'target_system', 1),
            getattr(vehicle, 'target_component', 0),
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            4,  # 4 Hz
            1   # Start
        )

        # Collect messages for the specified timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = vehicle.recv_match(blocking=False)
            if not msg:
                time.sleep(0.1)
                continue

            msg_type = msg.get_type()

            # Process message based on type
            if msg_type == "HEARTBEAT":
                diagnostics["heartbeat_received"] = True
                diagnostics["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                try:
                    if hasattr(mavutil, 'mode_string_v10') and callable(mavutil.mode_string_v10):
                        diagnostics["mode"] = mavutil.mode_string_v10(msg)
                    else:
                        diagnostics["mode"] = f"Mode ID: {msg.custom_mode}"
                except:
                    diagnostics["mode"] = f"Mode ID: {msg.custom_mode}"

            elif msg_type == "STATUSTEXT":
                message_text = msg.text if hasattr(msg, 'text') else "Unknown status"
                diagnostics["status_text_messages"].append(message_text)

                # Check for pre-arm status
                if "PreArm" in message_text:
                    diagnostics["pre_arm_status"].append(message_text)

            elif msg_type == "GPS_RAW_INT":
                diagnostics["gps_status"] = {
                    "fix_type": msg.fix_type,
                    "satellites_visible": msg.satellites_visible
                }

            elif msg_type == "AUTOPILOT_VERSION":
                # Extract version information
                flight_sw_version = msg.flight_sw_version
                major = (flight_sw_version >> 24) & 0xFF
                minor = (flight_sw_version >> 16) & 0xFF
                patch = (flight_sw_version >> 8) & 0xFF
                diagnostics["firmware_version"] = f"{major}.{minor}.{patch}"

            elif msg_type == "PARAM_VALUE":
                diagnostics["params_received"] = True

        return diagnostics
    except Exception as e:
        logging.error(f"Error getting vehicle diagnostics: {str(e)}")
        return None

# --- drone/connection.py ---
def reset_flight_controller(vehicle):
    """
    Attempt to reset the flight controller.

    Args:
        vehicle: The connected mavlink object

    Returns:
        True if reset command was sent successfully, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.warning("Sending reboot command to flight controller")

        # Send reboot command
        target_system = getattr(vehicle, 'target_system', 1)
        target_component = getattr(vehicle, 'target_component', 0)

        vehicle.mav.command_long_send(
            target_system,
            target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,                  # Confirmation
            1,                  # Param 1: 1=reboot autopilot
            0,                  # Param 2: 0=do nothing for onboard computer
            0,                  # Param 3: reserved
            0,                  # Param 4: reserved
            0, 0, 0             # Params 5-7 (not used)
        )

        logging.info("Reboot command sent. Wait for flight controller to restart.")
        return True
    except Exception as e:
        logging.error(f"Error sending reboot command: {str(e)}")
        return False
