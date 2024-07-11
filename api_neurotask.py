import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.signal import decimate
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

def get_dataframe(data, filter_result=False):
    bin = 1000/data.nwb.processing['spikes'].data_interfaces['spikes_counts'].rate
    
    keys = list(data.keys())
    dataframes = []
    
    for key in keys:
        if key == 'spikes_counts':
            # Create DataFrame for 'spikes_counts' with 'Neuron' prefix
            sp = pd.DataFrame(data['spikes_counts'].values, columns=data['spikes_counts'].columns)
            sp.columns = ['Neuron' + str(col) for col in sp.columns]
            dataframes.append(sp)
        else:
            df = pd.DataFrame(data[key].values, columns=[key])

            dataframes.append(df)
    
    
    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dataframes, axis=1)
    print(f'Data loaded with bin size of {bin:.1f} ms')

    if filter_result:
        return final_df[final_df['result'].isin(filter_result)], bin
    else:
        return final_df, bin

def load_and_filter_parquet(parquet_file_path, filter_letters = None):
    """
    Load a Parquet file, apply filters if provided, and return the filtered DataFrame and bin size.

    Parameters:
    parquet_file_path (str): Path to the Parquet file.
    filter_letters (list, optional): List of letters to filter out from the 'result' column.

    Returns:
    tuple: Filtered DataFrame and bin size as a float.
    """

    # Read the Parquet file with filters applied
    if filter_letters:
        table = pq.read_table(parquet_file_path, filters=[('result', '!=', letter) for letter in filter_letters])
    else:
        table = pq.read_table(parquet_file_path)

    # Convert the filtered table to a DataFrame
    df = table.to_pandas()

    # Extract the bin size from the file name
    bin = float(parquet_file_path.split('_')[1])



    print(f'Data loaded from {parquet_file_path} with bin size of {bin:.1f} ms')
    print('Events columns:',[col for col in df.columns if col.startswith('Event')])

    print('Covariates columns:', [col for col in df.columns if not col.startswith('Event') and not col.startswith('Neuron') and col not in ['trial_id','result','datasetID','session','animal', 'task']])

    return df, bin



def rebin(dataset1, prev_bin_size,new_bin_size, reset=True):
    """
    Rebin the given dataset to a new bin size.

    Parameters:
    dataset1 (pd.DataFrame): The dataset to rebin.
    prev_bin_size (int): The previous bin size.
    new_bin_size (int): The new bin size.
    reset (bool): Whether to reset the index and drop specific columns.

    Returns:
    pd.DataFrame: The rebinned dataset.
    """

    # Reset the index of the dataset
    d = dataset1.reset_index()

    # Calculate the bin size
    bin_size = new_bin_size//prev_bin_size

    # Group the dataset by session, trial_id, and the calculated bin size
    grouped = d.groupby(['session', 'trial_id', d.index // bin_size])
    agg_functions = {}

    # Define a safe decimation function to handle small data lengths
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

    # Aggregate the data based on the defined functions
    data_bin = grouped.agg(agg_functions)

    # Reset index
    if reset:
        del data_bin['session']
        del data_bin['trial_id']
        data_bin = data_bin.reset_index()
        del data_bin['level_2']

    return data_bin

def align_trial(df,start_event,bin_size,offset_min=None,offset_max=None):
    """
    Align trials in a DataFrame based on a start event and bin size.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        start_event (str): The column name indicating the start event.
        bin_size (int): The bin size of the data.
        offset_min (int, optional): The minimum offset for backward filling. Must be <= 0.
        offset_max (int, optional): The maximum offset for forward filling. Must be >= 0.
    
    Returns:
        pd.DataFrame: The DataFrame with aligned trials.
    """
    
    df[start_event] = df[start_event].replace(False, np.nan)
    df['ev'] = df[start_event]
    if offset_min:
        assert offset_min <= 0, "offset_min must be less than or equal to 0"
        offset_min=-offset_min//bin_size
        df['ev'] = df['ev'].bfill(limit=offset_min).infer_objects(copy=False)
    if offset_max:
        assert offset_max >= 0, "offset_max must be greater than or equal to 0"
        offset_max=offset_max//bin_size
        df['ev'] = df['ev'].ffill(limit=offset_max).infer_objects(copy=False)
    else:
        df['ev'] = df['ev'].ffill()
    df = df[(df['ev'] == 1)]
    del(df['ev'])
    return df

def align_event(df,start_event,bin_size,offset_min=None,offset_max=None):
    # Apply align_trial to each group and concatenate the results
    return pd.concat([align_trial(group, start_event,bin_size, offset_min,offset_max) for _, group in df.groupby(['animal','session','trial_id'], group_keys=True)], ignore_index=False)


###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.NaN
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1
    return X


def process_data(df, bins_before, training_range, valid_range, testing_range, behavior_columns,zscore=False):

    """
    Process the dataset, splitting it into training, validation, and testing sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        bins_before (int): Number of bins before the output used for decoding.
        training_range (list): The range [start, end] for the training set.
        valid_range (list): The range [start, end] for the validation set.
        testing_range (list): The range [start, end] for the testing set.
        behavior_columns (list): List of columns containing behavioral data.
        zscore (bool): Whether to apply z-score normalization. Defaults to False.

    Returns:
        tuple: A tuple containing lists of training, validation, and testing data.

    """

    neurons = [col for col in df.columns if col.startswith('Neuron')]
    X_train_list = []
    X_test_list = []
    X_val_list = []

    y_train_list = []
    y_test_list = []
    y_val_list = []

    # Iterate over each unique animal in the dataset
    for a in df['animal'].unique():
        # Select data for the current animal
        d = df[df['animal'] == a]

        # Iterate over each session for the current animal
        for session in d['session'].unique():
            # Select data for the current session and filter out zero columns
            df_session = df[(df['animal'] == a) & (df['session'] == session)][neurons].dropna(axis=1)
            df_session = df_session.loc[:, (df_session != 0).any(axis=0)]

            # Extract behavior data for the current session
            y = np.array(df[(df['session'] == session) & (df['animal'] == a)][behavior_columns])

            # Convert DataFrame to NumPy array
            session_data = df_session.to_numpy()

            # Get the covariate matrix that includes spike history from previous bins
            X = get_spikes_with_history(session_data, bins_before, 0, 1)
            num_examples = X.shape[0]

            # Define the ranges for training, testing, and validation sets
            training_set = np.arange(int(np.round(training_range[0] * num_examples)) + bins_before, int(np.round(training_range[1] * num_examples)))
            testing_set = np.arange(int(np.round(testing_range[0] * num_examples)) + bins_before, int(np.round(testing_range[1] * num_examples)) )
            valid_set = np.arange(int(np.round(valid_range[0] * num_examples)) + bins_before, int(np.round(valid_range[1] * num_examples)) )

            # Get training data
            X_train = X[training_set, :, :]
            y_train = y[training_set, :]

            # Get testing data
            X_test = X[testing_set, :, :]
            y_test = y[testing_set, :]

            # Get validation data
            X_valid = X[valid_set, :, :]
            y_valid = y[valid_set, :]

            if zscore:

                # Z-score "X" inputs
                X_train_mean = np.nanmean(X_train, axis=0)
                X_train_std = np.nanstd(X_train, axis=0)
                X_train_std = np.where(X_train_std == 0, 1e-16, X_train_std)

                X_train = (X_train - X_train_mean) / X_train_std
                X_test = (X_test - X_train_mean) / X_train_std
                X_valid = (X_valid - X_train_mean) / X_train_std

                # Zero-center outputs
                y_train_mean = np.mean(y_train, axis=0)
                y_train = y_train - y_train_mean
                y_test = y_test - y_train_mean
                y_valid = y_valid - y_train_mean

            X_train_list.append(X_train)
            X_test_list.append(X_test)
            X_val_list.append(X_valid)

            y_train_list.append(y_train)
            y_test_list.append(y_test)
            y_val_list.append(y_valid)

    return X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list

def process_data_forecast(df, bins_before, training_range, valid_range, testing_range,zscore=False):

    """
    Process the dataset, splitting it into training, validation, and testing sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        bins_before (int): Number of bins before the output used for decoding.
        training_range (list): The range [start, end] for the training set.
        valid_range (list): The range [start, end] for the validation set.
        testing_range (list): The range [start, end] for the testing set.
        behavior_columns (list): List of columns containing behavioral data.
        zscore (bool): Whether to apply z-score normalization. Defaults to False.

    Returns:
        tuple: A tuple containing lists of training, validation, and testing data.

    """

    neurons = [col for col in df.columns if col.startswith('Neuron')]
    X_train_list = []
    X_test_list = []
    X_val_list = []

    y_train_list = []
    y_test_list = []
    y_val_list = []

    # Iterate over each unique animal in the dataset
    for a in df['animal'].unique():
        # Select data for the current animal
        d = df[df['animal'] == a]

        # Iterate over each session for the current animal
        for session in d['session'].unique():
            # Select data for the current session and filter out zero columns
            df_session = df[(df['animal'] == a) & (df['session'] == session)][neurons].dropna(axis=1)
            df_session = df_session.loc[:, (df_session != 0).any(axis=0)]

            # Extract behavior data for the current session
            #y = np.array(df[(df['session'] == session) & (df['animal'] == a)][behavior_columns])

            # Convert DataFrame to NumPy array
            session_data = df_session.to_numpy()

            # Get the covariate matrix that includes spike history from previous bins
            X = get_spikes_with_history(session_data, bins_before, 0, 0)
            y = get_spikes_with_history(session_data,bins_before-1, 0, 1)
            num_examples = X.shape[0]

            # Define the ranges for training, testing, and validation sets
            training_set = np.arange(int(np.round(training_range[0] * num_examples)) + bins_before, int(np.round(training_range[1] * num_examples)))
            testing_set = np.arange(int(np.round(testing_range[0] * num_examples)) + bins_before, int(np.round(testing_range[1] * num_examples)) )
            valid_set = np.arange(int(np.round(valid_range[0] * num_examples)) + bins_before, int(np.round(valid_range[1] * num_examples)) )

            # Get training data
            X_train = X[training_set, :, :]
            y_train = y[training_set, :,:]

            # Get testing data
            X_test = X[testing_set, :, :]
            y_test = y[testing_set, :,:]

            # Get validation data
            X_valid = X[valid_set, :, :]
            y_valid = y[valid_set, :,:]

            if zscore:

                # Z-score "X" inputs
                X_train_mean = np.nanmean(X_train, axis=0)
                X_train_std = np.nanstd(X_train, axis=0)
                X_train_std = np.where(X_train_std == 0, 1e-16, X_train_std)

                X_train = (X_train - X_train_mean) / X_train_std
                X_test = (X_test - X_train_mean) / X_train_std
                X_valid = (X_valid - X_train_mean) / X_train_std

                # Zero-center outputs
                y_train_mean = np.mean(y_train, axis=0)
                y_train = y_train - y_train_mean
                y_test = y_test - y_train_mean
                y_valid = y_valid - y_train_mean

            X_train_list.append(X_train)
            X_test_list.append(X_test)
            X_val_list.append(X_valid)

            y_train_list.append(y_train)
            y_test_list.append(y_test)
            y_val_list.append(y_valid)

    return X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list

    # Define a function to scale a list of 3D arrays
def scale_data_list(data_list):
        scaler = MinMaxScaler()
        scaled_list = []
        for data in data_list:
            # Reshape to 2D for scaling
            original_shape = data.shape
            data_reshaped = data.reshape(-1, original_shape[-1])
            
            # Fit and transform the data
            data_scaled = scaler.fit_transform(data_reshaped)
            
            # Reshape back to 3D
            data_scaled_reshaped = data_scaled.reshape(original_shape)
            scaled_list.append(data_scaled_reshaped)
        return scaled_list, scaler

def process_data_forecast_mimo(df, bins_before,bins_after, training_range, valid_range, testing_range,filter_g=False, bin = 20, sigma = 50, scale = False):

    """
    Process the dataset, splitting it into training, validation, and testing sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        bins_before (int): Number of bins before the output used for decoding.
        training_range (list): The range [start, end] for the training set.
        valid_range (list): The range [start, end] for the validation set.
        testing_range (list): The range [start, end] for the testing set.
        behavior_columns (list): List of columns containing behavioral data.
        zscore (bool): Whether to apply z-score normalization. Defaults to False.

    Returns:
        tuple: A tuple containing lists of training, validation, and testing data.

    """

    neurons = [col for col in df.columns if col.startswith('Neuron')]
    X_train_list = []
    X_test_list = []
    X_val_list = []

    y_train_list = []
    y_test_list = []
    y_val_list = []

    # Iterate over each unique animal in the dataset
    for a in df['animal'].unique():
        # Select data for the current animal
        d = df[df['animal'] == a]

        # Iterate over each session for the current animal
        for session in d['session'].unique():
            # Select data for the current session and filter out zero columns
            df_session = df[(df['animal'] == a) & (df['session'] == session)][neurons].dropna(axis=1)
            df_session = df_session.loc[:, (df_session != 0).any(axis=0)]

            # Extract behavior data for the current session
            #y = np.array(df[(df['session'] == session) & (df['animal'] == a)][behavior_columns])

            # Convert DataFrame to NumPy array
            session_data = df_session.to_numpy()
            if filter_g:
                session_data = gaussian_filter(df_session.to_numpy(), sigma=sigma / bin)

            # Get the covariate matrix that includes spike history from previous bins
            X = get_spikes_with_history(session_data, bins_before, 0, 0)
            y = get_spikes_with_history(session_data,0, bins_after-1, 1)
            num_examples = X.shape[0]

            # Define the ranges for training, testing, and validation sets
            training_set = np.arange(int(np.round(training_range[0] * num_examples)) + bins_before, int(np.round(training_range[1] * num_examples)))
            testing_set = np.arange(int(np.round(testing_range[0] * num_examples)) + bins_before, int(np.round(testing_range[1] * num_examples)) )
            valid_set = np.arange(int(np.round(valid_range[0] * num_examples)) + bins_before, int(np.round(valid_range[1] * num_examples)) )

            # Get training data
            X_train = X[training_set, :, :]
            y_train = y[training_set, :,:]

            # Get testing data
            X_test = X[testing_set, :, :]
            y_test = y[testing_set, :,:]

            # Get validation data
            X_valid = X[valid_set, :, :]
            y_valid = y[valid_set, :,:]


            X_train_list.append(X_train)
            X_test_list.append(X_test)
            X_val_list.append(X_valid)

            y_train_list.append(y_train)
            y_test_list.append(y_test)
            y_val_list.append(y_valid)

    if scale:

        # Scale the input data lists
        X_train_list, X_scaler = scale_data_list(X_train_list)
        X_val_list, _ = scale_data_list(X_val_list)
        X_test_list, _ = scale_data_list(X_test_list)

        y_train_list, y_scaler = scale_data_list(y_train_list)
        y_val_list, _ = scale_data_list(y_val_list)
        y_test_list, _ = scale_data_list(y_test_list)

    return X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list