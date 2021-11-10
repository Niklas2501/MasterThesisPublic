import datetime as dt
import itertools
import json
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
from kafka import KafkaConsumer, TopicPartition

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from configuration.Configuration import Configuration


def import_kafka_topics():
    """
    loops through all the topics from the given Dataset with 'name', 'start_time' and 'end_time'

    The start and end time in config.json should be the exact start time of the first WF
    and the exact end time of the last WF
    """
    start_consume_time = datetime.now()

    print('-------------------------------')
    print('Importing datasets')
    print('start_time: {}'.format(datetime.now()))
    print('-------------------------------')

    config = Configuration()
    datasets_to_import = config.datasets_to_import_from_kafka
    kafka_topics_combined = list(set(
        list(config.kafka_webserver_topics.keys())
        + list(config.kafka_sensor_topics.keys())
        + list(config.kafka_txt_topics.keys())
        + list(config.kafka_failure_sim_topics.keys())
    ))

    for dataset_input in datasets_to_import:
        start_time = datetime.strptime(dataset_input[1], '%Y-%m-%d %H:%M:%S.%f') - dt.timedelta(0, 20)
        end_time = datetime.strptime(dataset_input[2], '%Y-%m-%d %H:%M:%S.%f') + dt.timedelta(0, 20)
        data_name = dataset_input[0]
        # -----------------------Test------------------------------
        # topic = 'Webserver_Log'
        # consume_data(topic, start_time, end_time, config, data_name)
        # ---------------------------------------------------------

        with Pool(processes=config.preprocessing_pool_size) as pool:
            pool.starmap(consume_data, zip(kafka_topics_combined,
                                           itertools.repeat(start_time),
                                           itertools.repeat(end_time),
                                           itertools.repeat(config),
                                           itertools.repeat(data_name), ))
        # add two columns to all TXT_topics with service_url and condition_url
        print("add two columns to all TXT_topics with service_url and condition_url from Webserver_Log")
        add_urls_to_txt_topics(dataset_input)

        # for topic in kafka_topics_combined:
        #     print(topic)
        #     consume_data(topic, start_time, end_time, config, data_name)
        l = os.listdir("../data/datasets/kafka_import/{}".format(data_name))
        if len(l) < 44:
            print(
                "------------------->>>> NOT ALL DATA COULD BE CONSUMED!!!!!!!!!! \n {} files could be consumed.".format(
                    len(l)))
        else:
            print("ALL {} FILES COULD BE CONSUMED!".format(len(l)))
    print('-------------------------------')
    final_time = datetime.now() - start_consume_time
    print('end_time: {}'.format(datetime.now()))
    print('time_to_consume: {}'.format(final_time))


def consume_data(topic: str, start_time: datetime, end_time: datetime, config: Configuration, data_name: str):
    """
    starts the consumer and writes the given topic to a .pkl file
    :param data_name: the name from the given dataset
    :param topic: the topic to consume
    :param start_time:
    :param end_time:
    :param config: the configuration to get specific information
    """
    # starts the consumer with the earliest offset
    consumer = KafkaConsumer(bootstrap_servers=[config.kafka_server_ip],
                             auto_offset_reset='earliest', group_id="data_import", enable_auto_commit=False)
    consumer.poll()

    # assign to the given topic and reset the offset
    tp = TopicPartition(topic, 0)
    consumer.assign([tp])
    consumer.seek_to_beginning(tp)

    # converts the given times to timestamp
    start_time = int(start_time.timestamp() * 1000)
    end_time = int(end_time.timestamp() * 1000)

    rec_in = consumer.offsets_for_times({tp: start_time})
    if rec_in is None or rec_in[tp] is None:
        print("No data at this start time! topic: {}".format(topic))
        return
    try:
        consumer.seek(tp, rec_in[tp].offset)
    except AttributeError:
        print("Attribute Error in Topic: {}".format(topic))
        return

    rec_out = consumer.offsets_for_times({tp: end_time})

    # set the offset to the the given time
    if rec_out[tp] is None:
        offset = consumer.end_offsets([tp])
        end_offset = offset[tp] - 1
    else:
        end_offset = rec_out[tp].offset
    start_offset = rec_in[tp].offset

    # print("import: {}".format(topic))
    # message count
    c = 0
    content_df = pd.DataFrame()
    # progress_bar
    pb = progress_bar(start_offset, end_offset)
    pb.start(topic)

    for msg in consumer:
        # update progress_bar
        pb.check_progress(topic)
        # bread if time_slot is consumed
        if msg.offset >= end_offset:
            pb.finish(topic)
            break
        # there are three formats how the topics are produced
        if topic == "Webserver_Log":
            content_df = content_df.append(import_webserver(msg))
            # c, content_df = import_webserver_log(consumer, start_offset, end_offset)
        elif "Jib" in topic or "Vibration" in topic:
            content_df = content_df.append(import_sensor(msg))
            # c, content_df = import_sensors(consumer, start_offset, end_offset)
        else:
            content_df = content_df.append(import_txt(msg))
            # c, content_df = import_txts(consumer, start_offset, end_offset)
        c += 1
        pb.increment_progress()

    consumer.close()
    if "Jib" in topic or "Vibration" in topic:
        content_df = content_df.rename(columns={'time': 'timestamp'})

    pb.final_message_count(topic, c)

    if content_df.empty:
        pb.nothing_to_save(topic)
        # print("empty Dataframe: nothing to save!")
    else:
        prefix = config.prefixes[topic.lower()]

        prefix = prefix + '_'
        content_df = content_df.add_prefix(prefix)
        content_df = content_df.rename(columns={prefix + 'timestamp': 'timestamp'})
        content_df.columns = content_df.columns.str.lower()

        # Remove lines with duplicate timestamps, keep first appearance and convert timestamp to pd.datetime and index
        content_df = content_df.loc[~content_df['timestamp'].duplicated(keep='first')]
        content_df['timestamp'] = pd.to_datetime(content_df['timestamp'])
        content_df = content_df.set_index('timestamp')
        content_df = content_df.sort_index()

        # Remove local fallback when only execution on server is relevant
        local_fallback = True
        if local_fallback:
            path = "../data/datasets/kafka_import/" + data_name + "/"
            if not os.path.exists(path):
                os.mkdir(path)
            content_df.to_pickle(path + topic + '_pkl')

        else:
            # Adapted for execution on gpu server using new data structure, untested
            dataset_path = config.kafka_imported_topics_path + '/{}/'.format(data_name)

            if not os.path.exists(dataset_path):
                os.mkdir(dataset_path)

            content_df.to_pickle(dataset_path + topic + '_pkl')

    #     print('Saving finished: {}'.format(topic))
    # print('-------------------------------')
    # print('-------------------------------')


def import_webserver(msg):
    df_webserver_log = pd.DataFrame()
    content = {}
    content_dict = dict(json.loads(str(msg.value, 'utf-8')))
    content.update(content_dict["service_parameters"])
    content.update(content_dict["execution_parameters"])
    del content_dict['service_parameters']
    del content_dict['execution_parameters']
    if "returned_attributes" in content_dict.keys():
        content.update(content_dict["returned_attributes"])
        del content_dict['returned_attributes']
    content.update(content_dict)
    content['timestamp'] = str(datetime.fromtimestamp(msg.timestamp / 1000))
    df_webserver_log = df_webserver_log.append(content, content.keys())
    return df_webserver_log


def import_sensor(msg) -> dict:
    return json.loads(str(msg.value, 'utf-8'))


def import_txt(msg) -> pd.DataFrame:
    df_txt = pd.DataFrame()
    content = json.loads(str(msg.value, 'utf-8'))
    content['timestamp'] = str(datetime.fromtimestamp(msg.timestamp / 1000))
    df_txt = df_txt.append(content, content.keys)
    return df_txt


def add_urls_to_txt_topics(dataset_input: list):
    """
    Gets a description of the recorded dataset and adds the Service_URL and Condition_URL
    columns to the respective TXT_Topic files.
    :type dataset_input: a list of strings like:  "failure_19_05_2021",
                                            "2021-05-19 10:36:38.035616",
                                            "2021-05-19 11:28:22.472997"
    """
    print("add service/condition urls to txt-topics for dataset: {}".format(dataset_input[0]))
    config = Configuration()
    all_txt_dataset_inputs = config.kafka_txt_topics

    # a list with all service_urls
    service_url_period = []
    # a list with all condition_urls
    condition_url_period = []
    webserver_df: pd.DataFrame = pd.read_pickle(
        config.kafka_imported_topics_path + f'{dataset_input[0]}/Webserver_Log_pkl')

    # collect the service and condition urls
    for index, row in webserver_df.iterrows():
        # if service_url (136...)
        # get start and end_time of the specific Service
        if "136" in row["webserver_log_full_url"].split('.')[0] and pd.isna(row["webserver_log_operation_end_time"]):
            start_time = webserver_df.at[index, "webserver_log_operation_start_time"]
            # creates a dataframe with 2 rows: request_url and response url
            df = webserver_df.loc[(webserver_df["webserver_log_operation_start_time"] == start_time) & (
                    webserver_df["webserver_log_full_url"] == row["webserver_log_full_url"])]
            if len(df.index) == 2:
                service_url_period.append(
                    {"start": [df.iloc[0]["webserver_log_full_url"], df.iloc[0]["webserver_log_resource"],
                               df.iloc[0]["webserver_log_operation_start_time"]],
                     "end": [df.iloc[1]["webserver_log_full_url"], df.iloc[0]["webserver_log_resource"],
                             df.iloc[1]["webserver_log_operation_start_time"],
                             df.iloc[1]["webserver_log_operation_end_time"]]})
        # if condition_url (127...)
        # get start and end_time of the specific Service
        if "127" in row["webserver_log_full_url"].split('.')[0] and pd.isna(row["webserver_log_operation_end_time"]):
            start_time = webserver_df.at[index, "webserver_log_operation_start_time"]
            # creates a dataframe with 2 rows: request_url and response url
            df = webserver_df.loc[(webserver_df["webserver_log_operation_start_time"] == start_time) & (
                    webserver_df["webserver_log_full_url"] == row["webserver_log_full_url"])]
            if len(df.index) == 2:
                condition_url_period.append(
                    {"start": [df.iloc[0]["webserver_log_full_url"], df.iloc[0]["webserver_log_resource"],
                               df.iloc[0]["webserver_log_operation_start_time"]],
                     "end": [df.iloc[1]["webserver_log_full_url"], df.iloc[0]["webserver_log_resource"],
                             df.iloc[1]["webserver_log_operation_start_time"],
                             df.iloc[1]["webserver_log_operation_end_time"]]})

    # a list of dictionaries with pairs of: resource: [services, ...]
    txt_topics_service = []
    # a list of dictionaries with pairs of: resource: [condition_services, ...]
    txt_topics_condition = []

    # filter the topics to get just the TXT-Modules
    for topic in all_txt_dataset_inputs:
        if len(topic.split("_")) == 2:
            txt_topics_service.append(topic)
            txt_topics_condition.append(topic)
    txt_topics_service = {k: [] for k in txt_topics_service}
    txt_topics_condition = {k: [] for k in txt_topics_condition}

    for elem in service_url_period:
        txt_topics_service[elem["start"][1].upper()].append(elem)

    for elem in condition_url_period:
        txt_topics_condition[elem["start"][1].upper()].append(elem)

    for e in txt_topics_service.items():
        resource = e[0]
        # load the resource_file
        df_1: pd.DataFrame = pd.read_pickle(
            config.kafka_imported_topics_path + f'{dataset_input[0]}/{resource}_pkl')
        df_1[resource.lower() + "_service_url"] = ".no_reading"
        df_1[resource.lower() + "_condition_url"] = ".no_reading"

        for value in e[1]:
            # get the nearest time_index from the resource_df
            dt_start = df_1.index.get_loc(get_timestamp(value["start"][2]), method='nearest')
            dt_end = df_1.index.get_loc(get_timestamp(value["end"][3]), method='nearest')
            # get all indices between start and end
            indices = df_1.loc[df_1.index[dt_start]:df_1.index[dt_end]].index
            # write url between start and end
            df_1.loc[indices, resource.lower() + "_service_url"] = value["start"][0]

        for value in txt_topics_condition[e[0]]:
            # get the nearest time_index from the resource_df
            dt_start = df_1.index.get_loc(get_timestamp(value["start"][2]), method='nearest')
            # get the index at the condition_time
            # index id datetime and index_int is the number in the "list"
            index = df_1.index[dt_start]
            index_int = dt_start
            # writes the condition_url to resource_condition_url at the condition_time where is no value
            while df_1.loc[df_1.index[index_int], resource.lower() + "_condition_url"] != ".no_reading":
                if index_int >= len(df_1.index) - 1:
                    break
                index = df_1.index[index_int + 1]
                index_int += 1
            df_1.loc[index, resource.lower() + "_condition_url"] = value["start"][0]
        # deletes the old file and saves the new file
        os.remove(config.kafka_imported_topics_path + f'{dataset_input[0]}/{resource}_pkl')
        df_1.to_pickle(config.kafka_imported_topics_path + f'{dataset_input[0]}/{resource}_pkl')
    print("Done!")


def get_timestamp(x: str) -> datetime:
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')


class progress_bar:
    def __init__(self, start_offset, end_offset):
        self.progress = start_offset
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.progress_one = (self.end_offset - self.start_offset - 1) / 20
        self.i = 0

    def start(self, topic):
        sys.stdout.write('\n')
        sys.stdout.write("[%-20s] %d%%" % ('=' * self.i, 5 * self.i))
        sys.stdout.write(" - {}".format(topic))
        sys.stdout.flush()
        self.i += 1

    def check_progress(self, topic):
        if self.progress >= self.start_offset + self.progress_one:
            self.progress = self.progress + self.progress_one
            sys.stdout.write('\n')
            sys.stdout.write("[%-20s] %d%%" % ('=' * self.i, 5 * self.i))
            sys.stdout.write(" - {}".format(topic))
            sys.stdout.flush()
            self.i += 1
            self.start_offset = self.progress
            return
        else:
            return

    def increment_progress(self):
        self.progress += 1

    def finish(self, topic):
        sys.stdout.write('\n')
        sys.stdout.write("[%-20s] %d%%" % ('=' * 20, 100))
        sys.stdout.write(" - {}".format(topic))
        sys.stdout.flush()
        # print("\n")

    def nothing_to_save(self, topic):
        sys.stdout.write('\n')
        sys.stdout.write("[%-20s] %d%%" % ('=' * 20, 100))
        sys.stdout.write(" - {} - Nothing to save!".format(topic))
        sys.stdout.flush()
        # print("\n")

    def final_message_count(self, topic, count):
        sys.stdout.write('\n')
        sys.stdout.write("[%-20s] %d%%" % ('=' * 20, 100))
        sys.stdout.write(" - {} - {} Messages where saved!".format(topic, str(count)))
        sys.stdout.flush()
        # print("\n")


if __name__ == '__main__':
    import_kafka_topics()
