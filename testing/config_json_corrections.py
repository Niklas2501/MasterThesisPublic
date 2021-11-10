import json
import os
import sys
import datetime

from datetime import  datetime as dt

from configuration.Configuration import Configuration

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

if __name__ == '__main__':
    config = Configuration()

    with open('../configuration/config.json', 'r') as f:
        data = json.load(f)

    for dataset, entries in data['run_to_failure_info'].items():
        for entry in entries:

            # Change end time of ov_1_measurement failures to be start_time + 5 seconds
            if "affected_component" in entry.keys() and entry["affected_component"] == "ov_1_measurement":
                start = dt.strptime(entry['start'], config.dt_format)
                entry["end"] = dt.strftime((start + datetime.timedelta(seconds=5)), config.dt_format)

            # Change RUL for all entries to be -1 if full failure instead of 0.0 and 0.01
            if "rul" in entry.keys() and entry["rul"] == 0.0:
                entry["rul"] = -1.0

            elif "rul" in entry.keys() and entry["rul"] == 0.01:
                entry["rul"] = 0.0

            if "affected_component" in entry.keys() and entry["affected_component"] == "sm_1_m1" and entry[
                "failure_mode"] in ['linear_type_1', 'linear_type_2']:
                entry["failure_mode"] = 'linear'

    with open('../configuration/config.json', 'w') as outfile:
        json.dump(data, outfile, sort_keys=False, indent=2)
