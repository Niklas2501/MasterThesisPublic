import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

def main():
    """
    Simple script to config.json entries with list or dictionary type values alphanumerically.
    """

    with open('../configuration/config.json', 'r') as f:
        data = json.load(f)

    list_type = ['integer_features', 'categorical_features', 'float_features', 'primary_features', 'secondary_features',
                 'unused_labels']
    dict_type = ['kafka_webserver_topics', 'kafka_sensor_topics', 'kafka_txt_topics', 'kafka_failure_simulation_topics',
                 'relevant_features', 'relevant_features_strict']

    for entry in list_type:
        data[entry] = sorted(data[entry])

    for entry in dict_type:
        data[entry] = dict(sorted(data[entry].items()))

    with open('../configuration/config_sorted.json', 'w') as outfile:
        json.dump(data, outfile, sort_keys=False, indent=2)


if __name__ == '__main__':
    main()
