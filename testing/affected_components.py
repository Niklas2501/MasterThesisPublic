import numpy as np
import pandas as pd

from configuration.Configuration import Configuration

config = Configuration()

affected_components = []
for dataset_entry in config.stored_datasets:
    dataset_name: str = dataset_entry[0]
    r2f_info = config.run_to_failure_info.get(dataset_name)
    df = pd.DataFrame.from_dict(r2f_info)

    if 'affected_component' in df.columns:
        affected_components.append(df['affected_component'])

affected_components = list(np.concatenate(affected_components))
affected_components = [ac for ac in affected_components if ac is not None]
affected_components, counts = np.unique(affected_components, return_counts=True)

for a, b in zip(affected_components, counts):
    print(a, b)
