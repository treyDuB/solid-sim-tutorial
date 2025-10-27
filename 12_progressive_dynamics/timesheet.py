import csv
import os

LOGFILE = "timesheet.csv"

def load_timesheet():
    """Load timesheet into a dict keyed by parameter tuple."""
    times = {}
    if os.path.exists(LOGFILE):
        with open(LOGFILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row[p] for p in reader.fieldnames[:-1])
                times[key] = float(row["time_seconds"])
    return times

def save_timesheet(times):
    """Write dict of times back to CSV."""
    fieldnames = ["level", "Big_L", "time_seconds"]
    with open(LOGFILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, t in times.items():
            writer.writerow({**dict(zip(fieldnames[:-1], k)), "time_seconds": t})
