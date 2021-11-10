import datetime as dt
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def label_failure():
    """
    Runs through all Failure_Simulation topics and looks for failure_visible.
    Then all free slots (without Failure_Simulation_visible = True) get marked too.
    Finally all time-slots between the dataset start- and end-time will be saved in config.json
    :return:
    """
    config = Configuration()
    data_to_write = False

    with open('../configuration/config.json', 'r') as f:
        data = json.load(f)
    # runs through each stored_dataset
    for dataset in data["stored_datasets"]:
        run_to_failure = []
        start_time = ""
        end_time = ""
        # if there are no run_to_failure_infos for this dataset yet a new list is created.
        if dataset[0] not in data["run_to_failure_info"].keys():
            print("dataset: {}".format(dataset[0]))
            data_to_write = True
            data["run_to_failure_info"][dataset[0]] = []
            failure_visibility = []

            # runs through all Failure_Simulation topics and writes the failure_slots in the new list
            # (failure_visibility)
            failure_sim_topics = data["kafka_failure_simulation_topics"]
            failure_sim_topics["OV_1_UV_Measurement"] = "OV_1_UV_Measurement"

            for topic in failure_sim_topics.keys():
                df_1: pd.DataFrame = pd.read_pickle(config.kafka_imported_topics_path + f'{dataset[0]}/{topic}_pkl')
                # updates the earliest start time for no_failure labeling
                if start_time != "" and start_time > df_1.index[0]:
                    start_time = df_1.index[0]
                elif start_time == "":
                    start_time = df_1.index[0]
                # updates the latest end time for no_failure labeling
                if end_time != "" and end_time < df_1.index[-1]:
                    end_time = df_1.index[-1]
                elif end_time == "":
                    end_time = df_1.index[-1]
                # searching in all failure topics for visible failures
                for c in df_1.columns:
                    # if a failure is visible
                    # writes all occurrences of a specific failure into a dataframe (all_df_change)
                    if "_visible" in c:
                        # get a dataframe with the indices where the visible value switches
                        all_df_change = pd.DataFrame()
                        all_df_change[c] = df_1[c].fillna(0)
                        all_df_change[c] = all_df_change[c].diff()
                        all_df_change = all_df_change.loc[(all_df_change != 0).any(1)]
                        all_df_change = all_df_change.loc[(all_df_change != 0).any(axis=1)]
                        all_df_change = all_df_change.dropna()
                        # if the failure is not visible continue to the next failure_visible column
                        if len(all_df_change) == 0:
                            continue

                        # get the last index inside the failure visible window. One before the actual index
                        # when the value drops from 1 to 0
                        dict_foo = {}
                        for i in all_df_change.index:
                            if all_df_change.at[i, c] == 1:
                                dict_foo[i] = 1
                            else:
                                if len(dict_foo) > 0 and df_1.iloc[df_1.index.get_loc(i) - 1].name in dict_foo.keys():
                                    pass
                                else:
                                    dict_foo[df_1.iloc[df_1.index.get_loc(i) - 1].name] = 0

                        # updates the visible_dataframe with the correct ent-time
                        all_df_change = pd.DataFrame.from_dict(dict_foo, orient='index', columns=[c])

                        # if the failure_run is not finished in this dataset -> end_of_failure is None
                        # the end-time of the recording is taken as the end_of_failure-time
                        end_of_failure = None
                        if len(all_df_change) > 0 and all_df_change.tail(1).index < df_1.tail(1).index:
                            end_of_failure = all_df_change.tail(1).index[0]

                        # get the name of the resource and manipulable
                        headline_split = str(c).split("_")
                        if topic == "OV_1_UV_Measurement":
                            resource = "OV_1_UV_Measurement"
                            manipulable = "ov_1_measurement"

                        else:
                            resource = headline_split[4] + "_" + headline_split[5]
                            manipulable = headline_split[4] + "_" + headline_split[5] + "_" + headline_split[6]
                        print("     resource: {}; manipulable: {}".format(resource, manipulable))

                        # add rul column to all_df_change dataframe
                        for col in df_1.columns:
                            if "remaining_useful_lifetime" in col and manipulable in col:
                                for index in all_df_change.index:
                                    all_df_change.at[index, "rul"] = df_1.at[index, col]

                        # get the failure_mode for this failure and manipulable
                        for x in df_1.columns:
                            if "failure_mode" in x and manipulable in x:
                                for i in all_df_change.index:
                                    all_df_change.at[i, "failure_mode"] = df_1.at[i, x]

                        # list_of_changes contains dfs with one run to failure
                        list_of_df_changes = []
                        for r_index, r in all_df_change["rul"].items():
                            actual_df_index = all_df_change.index.get_loc(r_index)
                            if len(all_df_change) - 1 == actual_df_index:
                                if len(list_of_df_changes) != 0 and list_of_df_changes[-1].index[-1] < r_index:
                                    l_change = all_df_change.iloc[
                                               all_df_change.index.get_loc(
                                                   list_of_df_changes[-1].index[-1]) + 1: actual_df_index + 1]
                                    list_of_df_changes.append(l_change)
                            elif r < all_df_change["rul"].iloc[all_df_change.index.get_loc(r_index) + 1]:
                                if len(list_of_df_changes) == 0:
                                    l_change = all_df_change.iloc[0: actual_df_index + 1]
                                else:
                                    l_change = all_df_change.iloc[
                                               all_df_change.index.get_loc(
                                                   list_of_df_changes[-1].index[-1]) + 1: actual_df_index + 1]
                                list_of_df_changes.append(l_change)

                        if len(list_of_df_changes) == 0:
                            list_of_df_changes.append(all_df_change)
                        for df_change in list_of_df_changes:
                            # cut the main df to the run_to_failure time (start- to end-time of the run)
                            df_1_slot = df_1.loc[df_change.index[0]: df_change.index[-1]]

                            # crates a list with tuples of the start- and end-time for all failure_slots in one run
                            failure_delta = []
                            for c_index, elem in df_change[c].items():
                                if elem == 1 and len(df_change.index) - 1 == df_change.index.get_loc(c_index):
                                    t = (c_index, df_1_slot.index[-1])
                                    failure_delta.append(t)
                                elif elem == 1:
                                    if df_change[c].iloc[df_change.index.get_loc(c_index) + 1] == 1:
                                        t = (c_index, c_index)
                                    else:
                                        t = (c_index, df_change.index[df_change.index.get_loc(c_index) + 1])
                                    failure_delta.append(t)

                            # failure time -> the smallest RUL in one run to failure
                            failure_time = "no failure time available"
                            for x in df_1_slot.columns:
                                if "remaining_useful_lifetime" in x and manipulable in x:
                                    df_failure_time = df_1_slot[x]
                                    failure_time = get_string(df_failure_time.idxmin())

                            # runs over all failure-slots
                            for i, delta in enumerate(failure_delta):
                                # all failure slots are being expanded +/- 2s
                                # because of the expansion it must be checked if the new slot overlaps an already
                                # existing slot. If yes the existing slot need to be updated

                                # get the WF which is currently running
                                if topic == "OV_1_UV_Measurement":
                                    df_resource: pd.DataFrame = pd.read_pickle(
                                        config.kafka_imported_topics_path + f'{dataset[0]}/{"OV_1"}_pkl')
                                    wf = df_resource.at[
                                        df_resource.index[df_resource.index.get_loc(delta[0] + dt.timedelta(0, 1),
                                                                                    "nearest")], "ov_1" +
                                        "_business_key"]
                                else:
                                    df_resource: pd.DataFrame = pd.read_pickle(
                                        config.kafka_imported_topics_path + f'{dataset[0]}/{resource.upper()}_pkl')

                                    wf = df_resource.at[
                                        df_resource.index[df_resource.index.get_loc(delta[0] + dt.timedelta(0, 1),
                                                                                    "nearest")], resource +
                                        "_business_key"]
                                if wf == "":
                                    wf = "null"

                                # if the failure_mode is an invert-Sensor-failure,
                                # some certain parameters will be added.
                                if df_change.at[delta[0], "failure_mode"] == "invert":
                                    # run to failure
                                    values = {"label": "failure",
                                              "failure_mode": "{}_{}".format(df_change.at[delta[0], "failure_mode"],
                                                                             float(int(df_change.at[delta[
                                                                                                        0], "rul"] * 1000) / 1000.0)),

                                              "affected_component": manipulable,
                                              "workflow": wf,
                                              "start": get_string(delta[0])}

                                    rul = "not available"
                                    for x in df_1_slot.columns:
                                        if "remaining_useful_lifetime" in x and manipulable in x:
                                            rul = df_1_slot.at[delta[1], x]
                                            rul = float(int(rul * 1000) / 1000.0)
                                    values["end"] = failure_time
                                    values["rul"] = rul
                                    values["failure_time"] = failure_time
                                    start_to_failure = str(get_timestamp(failure_time) - delta[0]).split("days")[1]
                                    from_failure_to_end = str(delta[1] - get_timestamp(failure_time)).split("days")[1]
                                    values["start_to_failure"] = start_to_failure
                                    failure_visibility.append(values)

                                    # failure to auto repair
                                    values = {"label": "failure",
                                              "failure_mode": "{}_failure".format(df_change.at[delta[0],
                                                                                               "failure_mode"]),
                                              "affected_component": manipulable,
                                              "workflow": wf,
                                              "start": failure_time}
                                    rul = "not available"
                                    for x in df_1_slot.columns:
                                        if "remaining_useful_lifetime" in x and manipulable in x:
                                            rul = df_1_slot.at[delta[1], x]
                                            rul = float(int(rul * 1000) / 1000.0)
                                    values["end"] = get_string(delta[1])
                                    values["rul"] = rul
                                    values["failure_time"] = failure_time
                                    from_failure_to_end = str(delta[1] - get_timestamp(failure_time)).split("days")[1]
                                    values["start_to_failure"] = start_to_failure
                                    values["from_failure_to_end"] = from_failure_to_end
                                    failure_visibility.append(values)
                                # update existing slot
                                # the timeslot window at start- and end-time is expanded by 500 milliseconds
                                elif len(failure_visibility) > 0 and len(failure_visibility[-1].keys()) > 2 and \
                                        get_timestamp(failure_visibility[-1]["end"]) > delta[0] and df_change.at[
                                    delta[0], "failure_mode"] == \
                                        failure_visibility[-1]["failure_mode"] and manipulable == \
                                        failure_visibility[-1][
                                            "affected_component"]:
                                    rul = "not available"
                                    for x in df_1_slot.columns:
                                        if "remaining_useful_lifetime" in x and manipulable in x:
                                            rul = df_1_slot.at[delta[1], x]
                                            rul = float(int(rul * 1000) / 1000.0)
                                    failure_visibility[-1]["end"] = get_string(
                                        delta[1] + dt.timedelta(milliseconds=500))
                                    failure_visibility[-1]["rul"] = rul
                                # creat new slot
                                # the timeslot window at start- and end-time is expanded by 500 milliseconds
                                else:
                                    values = {"label": "failure",
                                              "failure_mode": df_change.at[delta[0], "failure_mode"],
                                              "affected_component": manipulable,
                                              "workflow": wf,

                                              "start": get_string(delta[0] - dt.timedelta(milliseconds=500))}
                                    rul = "not available"
                                    for x in df_1_slot.columns:
                                        if "remaining_useful_lifetime" in x and manipulable in x:
                                            rul = df_1_slot.at[delta[1], x]
                                            rul = float(int(rul * 1000) / 1000.0)
                                    values["end"] = get_string(delta[1] + dt.timedelta(milliseconds=500))
                                    values["rul"] = rul
                                    values["failure_time"] = failure_time
                                    failure_visibility.append(values)
            # update run_to_failure with all failures from this topic
            run_to_failure.extend(failure_visibility)

        run_to_failure.sort(key=lambda item: datetime.strptime(item['start'], '%Y-%m-%d %H:%M:%S.%f'))

        # search for no_failure slots
        slots = []
        for i, e in enumerate(run_to_failure):
            values = {}
            end = "1990-05-19 00:00:00.000000"
            # last element in run_to_failure
            if i + 1 == len(run_to_failure):
                # sort for latest end-time
                run_to_failure.sort(key=lambda item: datetime.strptime(item['end'], '%Y-%m-%d %H:%M:%S.%f'))
                # if the latest end-time is smaller than the general end-time a new no_failure slot will be created
                if get_timestamp(run_to_failure[-1]["end"]) < end_time:
                    slots.append({"label": "no_failure",
                                  "start": run_to_failure[-1]["end"],
                                  "end": get_string(end_time)})
            # all elements between first- and last-element
            else:
                # first element in run_to_failure
                if i == 0:
                    if get_timestamp(e["start"]) > start_time:
                        slots.append({"label": "no_failure",
                                      "start": get_string(start_time),
                                      "end": e["start"]})
                out = False
                # all elements before the current one
                for c in run_to_failure[:i]:
                    # if the end-time parameter is bigger then the current end-time: there is nothing more to check
                    if get_timestamp(e["end"]) < get_timestamp(c["end"]):
                        out = True
                if not out:
                    # all elements after the current one
                    for c in run_to_failure[i + 1:]:
                        # if the start-time parameter is smaller then the current end-time parameter:
                        # there is nothing more to check
                        if get_timestamp(e["end"]) >= get_timestamp(c["start"]):
                            break
                        elif c is run_to_failure[-1]:
                            if end != "1990-05-19 00:00:00.000000":
                                slots.append({"label": "no_failure",
                                              "start": e["end"],
                                              "end": end})
                            else:
                                slots.append({"label": "no_failure",
                                              "start": e["end"],
                                              "end": c["start"]})
                        else:
                            if end == "1990-05-19 00:00:00.000000":
                                end = c["start"]
            if bool(values):
                slots.append(values)
        run_to_failure.extend(slots)
        run_to_failure.sort(key=lambda item: datetime.strptime(item['start'], '%Y-%m-%d %H:%M:%S.%f'))
        # if no failure in dataset
        if not run_to_failure:
            run_to_failure.extend([{"label": "no_failure",
                                    "start": dataset[1],
                                    "end": dataset[2]}])

        data["run_to_failure_info"][dataset[0]].extend(run_to_failure)

    if data_to_write:
        with open('../configuration/config.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("All Datasets are labeled!")
    else:
        print("No data to write!")


def get_timestamp(x: str) -> datetime:
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')


def get_string(x: datetime or pd.Index) -> str:
    return datetime.strftime(x, '%Y-%m-%d %H:%M:%S.%f')


if __name__ == '__main__':
    label_failure()
