"""
Drone Servo Control Module
------------------------
Functions for controlling servos for package delivery operations using pymavlink.
"""

import time
import logging
from pymavlink import mavutil

def set_servo_position(vehicle, servo_number, position):
    """
    Set a servo to a specific PWM position.

    Args:
        vehicle: The connected mavlink object
        servo_number: The servo number (1-16)
        position: PWM position (typically 1000-2000)

    Returns:
        True if successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Send DO_SET_SERVO command
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,                  # Confirmation
            servo_number,       # Param 1: Servo number
            position,           # Param 2: PWM position
            0, 0, 0, 0, 0       # Params 3-7 (not used)
        )

        # Wait for acknowledgment
        start_time = time.time()
        while time.time() - start_time < 3:  # 3 second timeout
            msg = vehicle.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_SERVO:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logging.info(f"Servo {servo_number} set to position {position}")
                    return True
                else:
                    logging.warning(f"Servo command failed with result {msg.result}")
                    return False

        logging.warning("No acknowledgment received for servo command")
        # We still return True as some autopilots don't send ACK for servo commands
        logging.info(f"Servo {servo_number} set to position {position} (no ACK)")
        return True
    except Exception as e:
        logging.error(f"Error setting servo position: {str(e)}")
        return False

def operate_package_release(vehicle, servo_number=9):
    """
    Release the package by operating the release servo.

    Args:
        vehicle: The connected mavlink object
        servo_number: The servo number for the release mechanism

    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("Operating package release mechanism")

        # First position - closed
        if not set_servo_position(vehicle, servo_number, 1000):
            logging.error("Failed to set initial servo position")
            return False
        time.sleep(1)

        # Second position - open to release package
        if not set_servo_position(vehicle, servo_number, 2000):
            logging.error("Failed to open release mechanism")
            return False
        time.sleep(2)

        # Return to closed position
        if not set_servo_position(vehicle, servo_number, 1000):
            logging.error("Failed to close release mechanism")
            return False

        logging.info("Package release completed")
        return True
    except Exception as e:
        logging.error(f"Error during package release: {str(e)}")
        return False

def operate_claw(vehicle, servo_number=10, open_position=2000, closed_position=1000):
    """
    Operate the claw for package delivery.

    Args:
        vehicle: The connected mavlink object
        servo_number: The servo number for the claw
        open_position: PWM value for open position
        closed_position: PWM value for closed position

    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("Opening claw")
        if not set_servo_position(vehicle, servo_number, open_position):
            logging.error("Failed to open claw")
            return False
        time.sleep(2)

        logging.info("Closing claw")
        if not set_servo_position(vehicle, servo_number, closed_position):
            logging.error("Failed to close claw")
            return False

        logging.info("Claw operation completed")
        return True
    except Exception as e:
        logging.error(f"Error operating claw: {str(e)}")
        return False

def test_servo(vehicle, servo_number, min_position=1000, max_position=2000, steps=5, step_time=1):
    """
    Test a servo by moving it through a range of positions.

    Args:
        vehicle: The connected mavlink object
        servo_number: The servo number to test
        min_position: Minimum PWM position
        max_position: Maximum PWM position
        steps: Number of steps between min and max
        step_time: Time to hold each position in seconds

    Returns:
        True if test was successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        logging.info(f"Testing servo {servo_number} from {min_position} to {max_position} PWM")

        # Calculate step size
        step_size = (max_position - min_position) // (steps - 1) if steps > 1 else 0

        # Move servo through each position
        for i in range(steps):
            position = min_position + (i * step_size)
            logging.info(f"Setting servo {servo_number} to position {position}")

            if not set_servo_position(vehicle, servo_number, position):
                logging.error(f"Failed to set servo to position {position}")
                return False

            time.sleep(step_time)

        # Return to neutral position
        neutral_position = (min_position + max_position) // 2
        logging.info(f"Returning servo {servo_number} to neutral position {neutral_position}")
        set_servo_position(vehicle, servo_number, neutral_position)

        logging.info(f"Servo {servo_number} test completed")
        return True
    except Exception as e:
        logging.error(f"Error during servo test: {str(e)}")
        return False

def set_servo_output_channel(vehicle, channel, output):
    """
    Set a raw servo output value.

    Args:
        vehicle: The connected mavlink object
        channel: The output channel number (0-based)
        output: The output value (typically 1000-2000)

    Returns:
        True if successful, False otherwise
    """
    if not vehicle:
        logging.error("No vehicle connection")
        return False

    try:
        # Convert channel to 1-based for the MAVLink command
        servo_number = channel + 1

        # Use DO_SET_SERVO command
        vehicle.mav.command_long_send(
            vehicle.target_system,
            vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,                  # Confirmation
            servo_number,       # Param 1: Servo number (1-based)
            output,             # Param 2: Output value
            0, 0, 0, 0, 0       # Params 3-7 (not used)
        )

        logging.info(f"Set channel {channel} to output {output}")
        return True
    except Exception as e:
        logging.error(f"Error setting servo output: {str(e)}")
        return False
