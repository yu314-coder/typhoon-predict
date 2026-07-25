#!/bin/bash
# Stop the ERA5 0.25-degree pull cleanly. Safe at ANY time:
#  - every COMPLETED year (era5_YYYY.npz) is already saved and intact (atomic write).
#  - the year in progress is discarded and simply re-done on resume (year granularity).
pkill -f extract_era5
sleep 2
n=$(pgrep -f extract_era5 | wc -l | tr -d ' ')
if [ "$n" = "0" ]; then
  echo "ERA5 pull STOPPED. Completed years are saved. Run ./era5_resume.sh to continue."
  ls track_build/era5/*.npz 2>/dev/null | sed -E 's/.*era5_([0-9]+).*/\1/' | sort -n | tr '\n' ' '; echo
else
  echo "WARNING: $n process(es) still alive; retry ./era5_stop.sh"
fi
