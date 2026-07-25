#!/bin/bash
# Resume (or start) the ERA5 0.25-degree training-year pull. Idempotent: completed years are
# skipped, corrupt/partial years are re-extracted, so this is safe to run repeatedly and after
# a reboot. Two processes split the range for parallelism.
cd "$(dirname "$0")"
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
# Force ALL temp scratch onto disk D too -- the main disk is tight (~12 GB). Without this, TMPDIR
# defaults to /var/folders/... on the main disk, where a large DAP spool could land.
export TMPDIR="$(pwd)/track_build/tmp/"
mkdir -p "$TMPDIR"
if pgrep -f extract_era5 >/dev/null; then
  echo "already running ($(pgrep -f extract_era5 | wc -l | tr -d ' ') procs). Stop first with ./era5_stop.sh"; exit 0
fi
nohup env MODE=full WORKERS=4 Y0=2005 YMAX=2019 REVERSE=1 "$PY" -W ignore extract_era5.py >> track_build/era5_recent.log 2>&1 &
echo "recent (2005-2019) resumed, PID $!"
sleep 2
# training set starts at 2001 -- 1990-2000 was dropped deliberately (user call, 2026-07-25)
nohup env MODE=full WORKERS=4 Y0=2001 YMAX=2004 REVERSE=0 "$PY" -W ignore extract_era5.py >> track_build/era5_old.log 2>&1 &
echo "older (2001-2004) resumed, PID $!"
echo "monitor: tail -f track_build/era5_recent.log track_build/era5_old.log"
