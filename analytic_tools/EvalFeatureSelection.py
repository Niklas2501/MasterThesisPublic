import argparse

import pandas
import pandas as pd


def fs_eval_step_3(sort_by, path_combined_log_file):

    sort_by = [sort_by]
    df_lines = []
    rank_count = 0
    baseline_found = False

    with open(path_combined_log_file) as fp:
        while True:

            line = fp.readline()

            if line == "\n" or not line:
                rank_count = 0
                baseline_found = False

                if not line:
                    break
            else:
                parts = line.split()
                group = parts.pop(-1)

                if group == 'baseline':
                    baseline_found = True

                before_baseline = 1 if not baseline_found else 0
                df_line = [group, rank_count, before_baseline, 1] + parts[1:]
                df_lines.append(df_line)

                rank_count += 1

    asc = [True] + [False if col not in ['Rank'] else True for col in sort_by]
    all = df_from_list(df_lines)
    all = all.sort_values(by=['Group'] + sort_by, ascending=asc)
    print(all.to_string(), '\n')

    process_df(all, sort_by)


def df_from_list(df_lines):
    df = pd.DataFrame(columns=['Group', 'Rank', 'BeforeBaseline', 'TimesTested', 'F1', 'F2', 'Recall', 'Precision'],
                      data=df_lines)

    return df


def process_df(df, sort_by: [], description=None):
    for col in ['F1', 'F2', 'Recall', 'Precision']:
        df[col] = pd.to_numeric(df[col])

    temp = df.groupby(by="Group").mean()
    # temp['BeforeBaseline'] = df.groupby(by="Group").sum()['BeforeBaseline']
    temp['TimesTested'] = df.groupby(by="Group").sum()['TimesTested']
    asc = [False if col not in ['Rank'] else True for col in sort_by]
    df = temp.sort_values(by=sort_by, ascending=asc)

    if description is not None:
        print(description)
    print(df.to_string(), '\n')

    return df


def fs_eval_step_2(sort_by):
    path_combined_log_file = '../logs_archive/feature_selection_step_2/fs_step_2_results_combined.log'
    log_names_ordered = [
        "08-12_feature_selection_2-4_15071936",
        "08-10_feature_selection_2-4_25011997",
        "08-10_feature_selection_4-6_15071936",
        "08-10_feature_selection_4-6_25011997",
        "08-10_feature_selection_6-8_15071936",
        "08-12_feature_selection_6-8_25011997",
    ]

    sort_by_individual = [sort_by]
    sort_by_all = [sort_by]

    current_df_lines = []
    rank_count = 0
    result_dfs = []
    baseline_found = False

    with open(path_combined_log_file) as fp:
        while True:

            line = fp.readline()

            if line == "\n" or not line:
                df = df_from_list(current_df_lines)

                if not df.empty:
                    result_dfs.append(df)

                rank_count = 0
                current_df_lines = []
                baseline_found = False

                if not line:
                    break
            else:
                parts = line.split()
                groups = parts.pop(-1).split('-')

                # Must be changed if individual groups were tested
                if len(groups) == 1:
                    if groups[0] == 'baseline':
                        baseline_found = True
                    else:
                        raise ValueError()

                for group in groups:
                    before_baseline = 1 if not baseline_found else 0
                    df_line = [group, rank_count, before_baseline, 1] + parts[1:]
                    current_df_lines.append(df_line)

                rank_count += 1

    grouped = []

    for i, df in enumerate(result_dfs):
        df = process_df(df, sort_by_individual, description=log_names_ordered[i])
        grouped.append(df)

    all = pandas.concat(grouped)
    process_df(all, sort_by_all, '#########' * 5)


if __name__ == '__main__':
    """
    Script that can be used for evaluating feature and a_pre selection runs. 
    """

    cols = ['Rank', 'BeforeBaseline', 'TimesTested', 'F1', 'F2', 'Recall', 'Precision']

    parser = argparse.ArgumentParser()
    parser.add_argument('--step',  choices=['2','3', 'a_variant'], required=True)
    parser.add_argument('--sort_by', choices=cols, default='F1')

    args, _ = parser.parse_known_args()

    if args.step == '2':
        fs_eval_step_2(sort_by=args.sort_by)
    elif args.step == '3':
        log_file = '../logs_archive/feature_selection_step_3/fs_step_3_results_combined.log'
        fs_eval_step_3(sort_by=args.sort_by, path_combined_log_file=log_file)
    elif args.step == 'a_variant':
        log_file = '../logs_archive/a_pre_selection/a_pre_selection_results_combined.log'
        fs_eval_step_3(sort_by=args.sort_by, path_combined_log_file=log_file)