import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.signal import decimate

def load_and_filter_parquet(parquet_file_path, filter_letters = None):
    # Read the Parquet file with filters applied
    if filter_letters:
        table = pq.read_table(parquet_file_path, filters=[('result', '!=', letter) for letter in filter_letters])
    else:
        table = pq.read_table(parquet_file_path)

    # Convert the filtered table to a DataFrame
    df = table.to_pandas()

    return df


def rebin(dataset1, bin_size, reset=False):
    d = dataset1.reset_index()

    grouped = d.groupby(['session', 'trial_id', d.index // bin_size])
    agg_functions = {}

    def safe_decimate(x, bin_size):
        if len(x) <= 27:
            return np.mean(x)
        return decimate(x, bin_size, ftype='iir', zero_phase=True).mean()

    # Define aggregation functions
    for col in dataset1.columns:
        if col.startswith('Neuron'):
            agg_functions[col] = 'sum'
        elif col.startswith('force') or col.startswith('hand') or col.startswith('finger') or col.startswith('cursor'):
            agg_functions[col] = lambda x: safe_decimate(x, bin_size)
        else:
            agg_functions[col] = 'max'

    data_bin = grouped.agg(agg_functions)
    
    if reset:
        del data_bin['session']
        del data_bin['trial_id']
        data_bin = data_bin.reset_index()
        del data_bin['level_2']
    
    return data_bin

def align_trial(df,start_event,offset_min=None,offset_max=None):
    df[start_event] = df[start_event].replace(0, np.nan)
    df['ev'] = df[start_event]
    if offset_min:
        df['ev'] = df['ev'].bfill(limit=offset_min).infer_objects(copy=False)
    if offset_max:
        df['ev'] = df['ev'].ffill(limit=offset_max).infer_objects(copy=False)
    else:
        df['ev'] = df['ev'].ffill()
    df = df[(df['ev'] == 1)]
    del(df['ev'])
    return df

def align_event(df,start_event,offset_min=None,offset_max=None):
    return pd.concat([align_trial(group, start_event, offset_min,offset_max) for _, group in df.groupby(['animal','session','trial_id'], group_keys=True)], ignore_index=False)
