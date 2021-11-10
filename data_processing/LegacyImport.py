import gc
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from configuration.Enums import Dataset as DSE


def import_txt(filename: str, prefix: str):
    # Load from file and transform into a json object.
    with open(filename) as f:
        content = json.load(f)

    # Transform into dataframe.
    df = pd.DataFrame.from_records(content)

    # Special case for txt controller 18 which has a sub message containing the position of the crane.
    # Split position column into 3 columns containing the x,y,z position.
    if '18' in prefix:
        pos = df['currentPos'].apply(lambda x: dict(eval(x.strip(','))))
        df['vsg_x'] = pos.apply(lambda r: (r['x'])).values
        df['vsg_y'] = pos.apply(lambda r: (r['y'])).values
        df['vsg_z'] = pos.apply(lambda r: (r['z'])).values
        df = df.drop('currentPos', axis=1)

    # Add the prefix to every column except the timestamp
    prefix = prefix + '_'
    df = df.add_prefix(prefix)
    df = df.rename(columns={prefix + 'timestamp': 'timestamp'})

    # Remove lines with duplicate timestamps, keep first appearance.
    df = df.loc[~df['timestamp'].duplicated(keep='first')]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    return df


def import_all_txts(query, raw_data_path):
    print('Importing TXT controller data')

    # Import the single txt sensors.
    df15: pd.DataFrame = import_txt(raw_data_path + 'txt15.txt', 'txt15')
    df16: pd.DataFrame = import_txt(raw_data_path + 'txt16.txt', 'txt16')
    df17: pd.DataFrame = import_txt(raw_data_path + 'txt17.txt', 'txt17')
    df18: pd.DataFrame = import_txt(raw_data_path + 'txt18.txt', 'txt18')
    df19: pd.DataFrame = import_txt(raw_data_path + 'txt19.txt', 'txt19')

    # Combine into a single dataframe.
    df_txt = df15.join(df16, how='outer')
    df_txt = df_txt.join(df17, how='outer')
    df_txt = df_txt.join(df18, how='outer')
    df_txt = df_txt.join(df19, how='outer')

    df_txt.query(query, inplace=True)

    return df_txt


def import_single_pressure_sensor(content, c_name: str, module):
    record_list = []

    # Add prefixes to message components.
    for i in content:
        temp = i[c_name]
        temp['hPa_' + module] = temp.pop('hPa')
        temp['tC_' + module] = temp.pop('tC')
        temp['timestamp'] = i['meta']['time']
        record_list.append(temp)

    df = pd.DataFrame.from_records(record_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    return df


def import_pressure_sensors(query, raw_data_path):
    print('\nImporting pressure sensors')

    with open(raw_data_path + 'pressureSensors.txt') as f:
        content = json.load(f)

    # Import the single components of the message.
    df_pres_18 = import_single_pressure_sensor(content, 'VSG', '18')
    df_pres_17 = import_single_pressure_sensor(content, 'Oven', '17')
    df_pres_15 = import_single_pressure_sensor(content, 'Sorter', '15')

    # combine into a single dataframe
    df_sensor_data = df_pres_18.merge(df_pres_17, left_on='timestamp', right_on='timestamp', how='inner')
    df_sensor_data = df_sensor_data.merge(df_pres_15, left_on='timestamp', right_on='timestamp', how='inner')

    # Change format of timestamp, set it as index and reduce the time interval.
    df_sensor_data['timestamp'] = pd.to_datetime(df_sensor_data['timestamp'])
    df_sensor_data = df_sensor_data.set_index(df_sensor_data['timestamp'])
    df_sensor_data.query(query, inplace=True)
    df_sensor_data.drop('timestamp', 1, inplace=True)

    return df_sensor_data


def import_acc(filename: str, prefix: str):
    print('Importing ' + filename)

    # Load from file and transform into a json object.
    with open(filename) as f:
        content = json.load(f)

    entry_list = []

    # Extract single messages and add prefixes to the message entries.
    for m in content:

        for e in m:
            e[prefix + '_x'] = e.pop('x')
            e[prefix + '_y'] = e.pop('y')
            e[prefix + '_z'] = e.pop('z')

            # Partly different naming.
            if 'timestamp' not in e.keys():
                e['timestamp'] = e.pop('time')

            entry_list.append(e)

    df = pd.DataFrame.from_records(entry_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    return df


def import_acc_sensors(query, raw_data_path):
    print('\nImport acceleration sensors')

    # Import each acceleration sensor.
    acc_txt15_m1 = import_acc(raw_data_path + 'TXT15_m1_acc.txt', 'a_15_1')
    acc_txt15_comp = import_acc(raw_data_path + 'TXT15_o8Compressor_acc.txt', 'a_15_c')
    acc_txt16_m3 = import_acc(raw_data_path + 'TXT16_m3_acc.txt', 'a_16_3')
    acc_txt18_m1 = import_acc(raw_data_path + 'TXT18_m1_acc.txt', 'a_18_1')

    # Combine into a single dataframe.
    acc_txt15_m1['timestamp'] = pd.to_datetime(acc_txt15_m1['timestamp'])
    df_accs = acc_txt15_m1.set_index('timestamp').join(acc_txt15_comp.set_index('timestamp'), how='outer')
    df_accs = df_accs.join(acc_txt16_m3.set_index('timestamp'), how='outer')
    df_accs = df_accs.join(acc_txt18_m1.set_index('timestamp'), how='outer')
    df_accs.query(query, inplace=True)

    return df_accs


def import_bmx_acc(filename: str, prefix: str):
    print('Importing ' + filename)

    # Load from file and transform into a json object.
    with open(filename) as f:
        content = json.load(f)

    entry_list = []

    # Extract single messages and add prefixes to the message entries.
    for m in content:
        for e in m:
            e[prefix + '_x'] = e.pop('x')
            e[prefix + '_y'] = e.pop('y')
            e[prefix + '_z'] = e.pop('z')
            e[prefix + '_t'] = e.pop('t')
            # e['timestamp'] = e.pop('time')
            entry_list.append(e)

    # Transform into a data frame and return.
    df = pd.DataFrame.from_records(entry_list)
    df = df.loc[~df['timestamp'].duplicated(keep='first')]
    return df


def import_bmx_sensors(query, raw_data_path):
    print('\nImport bmx sensors')

    # import single components
    # df_hrs_acc = import_bmx_acc(config.bmx055_HRS_acc, 'hrs_acc')
    df_hrs_gyr = import_acc(raw_data_path + 'bmx055-HRS-gyr.txt', 'hrs_gyr')
    df_hrs_mag = import_acc(raw_data_path + 'bmx055-HRS-mag.txt', 'hrs_mag')

    # combine into a single dataframe
    df_hrs_gyr['timestamp'] = pd.to_datetime(df_hrs_gyr['timestamp'])
    df_hrs = df_hrs_gyr.set_index('timestamp').join(df_hrs_mag.set_index('timestamp'), how='outer')
    # df_hrs_gyr['timestamp'] = pd.to_datetime(df_hrs_gyr['timestamp'])
    # df_hrs = df_hrs.set_index('timestamp').join(df_hrs_mag.set_index('timestamp'), how='outer')
    df_hrs.query(query, inplace=True)

    # import single components
    df_vsg_acc = import_bmx_acc(raw_data_path + 'bmx055-VSG-acc.txt', 'vsg_acc')
    df_vsg_gyr = import_acc(raw_data_path + 'bmx055-VSG-gyr.txt', 'vsg_gyr')
    df_vsg_mag = import_acc(raw_data_path + 'bmx055-VSG-mag.txt', 'vsg_mag')

    # combine into a single dataframe
    df_vsg_acc['timestamp'] = pd.to_datetime(df_vsg_acc['timestamp'])
    df_vsg = df_vsg_acc.set_index('timestamp').join(df_vsg_gyr.set_index('timestamp'), how='outer')
    df_vsg = df_vsg.join(df_vsg_mag.set_index('timestamp'), how='outer')
    df_vsg.query(query, inplace=True)

    return df_hrs, df_vsg


def import_datasets():
    """
    Imports the given dataset and saves the combined dataframe as a .pkl in the directory given in the configuration.
    """
    config = Configuration()
    assert config.dataset == DSE.FT_LEGACY, 'Can only be used for the legacy dataset.'

    for dataset_name, start, end in config.stored_datasets:
        print(f'Started import for {dataset_name} for range {start} - {end}')

        query = "timestamp <= \'" + end + "\' & timestamp >= \'" + start + "\' "
        raw_data_path = config.kafka_imported_topics_path + f'{dataset_name}/raw_data/'
        storage_path = config.unprocessed_data_path + f'{dataset_name}/'

        # import each senor type
        df_txt_combined = import_all_txts(query, raw_data_path)
        df_press_combined = import_pressure_sensors(query, raw_data_path)
        df_accs_combined = import_acc_sensors(query, raw_data_path)
        df_hrs_combined, df_vsg_combined = import_bmx_sensors(query, raw_data_path)
        gc.collect()

        df_combined = df_press_combined.join(df_accs_combined, how='outer')
        df_combined = df_combined.join(df_hrs_combined, how='outer')
        df_combined = df_combined.join(df_vsg_combined, how='outer')

        df_txt_combined.query(query, inplace=True)
        df_combined.query(query, inplace=True)

        if not os.path.exists(storage_path):
            os.mkdir(storage_path)

        print('\nSaving dataframe as pickle file in', )
        df_txt_combined.to_pickle(storage_path + 'txt_topics.pkl')
        df_combined.to_pickle(storage_path + 'sensor_topics.pkl')
        print('Saving finished')

        print(*list(df_combined.columns.values), sep="\n")

        print('\nImporting of dataset', dataset_name, 'finished')


if __name__ == '__main__':
    import_datasets()
