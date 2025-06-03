# Voltage Sag
When on use the drone drops voltage more than the voltage actually went down once the flight is over.

## Voltage Sag Compensation
The Flight Controller has the ability to compensate for this voltage sag, but it might not be turned on by default:
- `BATT_FS_VOLTSRC`: This setting is required to be set to (1)[Sag Compensated Voltage] rather than (0)[Raw Voltage]

Additionally these settings might also need to be set for it to work
```
MOT_BAT_VOLT_MAX = 16.8    # (4.2V × 4 cells = 16.8V)
MOT_BAT_VOLT_MIN = 13.2    # (3.3V × 4 cells = 13.2V)
MOT_BAT_IDX = 0            # Which battery monitor to use
```
