import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from datetime import datetime as dt

from configuration.Configuration import Configuration


def get_conflicts(nf_start, nf_end, failure_intervals):
    conflicting_intervals = []

    for start, end, failure_interval in failure_intervals:

        if start < nf_start < end or start < nf_end < end or nf_start < start < nf_end or nf_start < end < nf_end:
            conflicting_intervals.append(failure_interval)

    return conflicting_intervals


def check_timestamp_format(r2f_entry, config):
    try:
        dt.strptime(r2f_entry['start'], config.dt_format)
        dt.strptime(r2f_entry['end'], config.dt_format)

        if 'failure_time' in r2f_entry.keys():
            dt.strptime(r2f_entry['failure_time'], config.dt_format)

        return False

    except ValueError:
        return True


def check_timestamp_order(r2f_entry, config):
    start = dt.strptime(r2f_entry['start'], config.dt_format)
    end = dt.strptime(r2f_entry['end'], config.dt_format)
    return end < start


def check_entries(r2f_info, entry_function, description, config):
    found_error = False

    for dataset, entries in r2f_info.items():

        entries_with_error = []

        for r2f_entry in entries:
            if entry_function(r2f_entry, config):
                entries_with_error.append(r2f_entry)

        if len(entries_with_error) > 0:
            print('-------------------------------------------------------------')
            print(f'{description} errors in {dataset}:')
            print('-------------------------------------------------------------')
            for entry in entries_with_error:
                print(json.dumps(entry, sort_keys=False, indent=4))
            print()
            found_error = True

    return found_error


def main():
    """
    Script that can be used to check the run_to_failure_info in the config.json for errors, mainly resulting from
    manual corrections. Will check the correctness of timestamp format and order and ensures no intervals overlap.
    """

    config = Configuration()
    r2f_info = config.run_to_failure_info

    # First check if the format of all timestamps is correct.
    found_error = check_entries(r2f_info, entry_function=check_timestamp_format, description='Timestamp format',
                                config=config)

    # If any timestamp has a wrong format, don't run any other check which would crash because of that.
    if found_error:
        sys.exit()

    check_entries(r2f_info, entry_function=check_timestamp_order, description='Start/End', config=config)

    for dataset, entries in r2f_info.items():
        ds_printed = False

        extracted_failure_intervals = []
        for r2f_entry in entries:
            if r2f_entry['label'] == 'failure':
                start = dt.strptime(r2f_entry['start'], config.dt_format)
                end = dt.strptime(r2f_entry['end'], config.dt_format)
                extracted_failure_intervals.append((start, end, r2f_entry))

        for r2f_entry in entries:

            if r2f_entry['label'] == 'no_failure':
                start = dt.strptime(r2f_entry['start'], config.dt_format)
                end = dt.strptime(r2f_entry['end'], config.dt_format)

                conflicting_intervals = get_conflicts(start, end, extracted_failure_intervals)

                if len(conflicting_intervals) > 0:

                    if not ds_printed:
                        print('-------------------------------------------------------------')
                        print(f'NF Interval consistency errors in {dataset}:')
                        print('-------------------------------------------------------------')
                        ds_printed = True

                    print(f'NF Interval: start = {start}, end = {end} \n')
                    print('Conflicts:')
                    for conflict in conflicting_intervals:
                        print(json.dumps(conflict, sort_keys=False, indent=4))
                    print()

    print('Checks finished.')


if __name__ == '__main__':
    main()
