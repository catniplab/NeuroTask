# The NeuroTask Benchmark Dataset

NeuroTask is a benchmark dataset designed to facilitate the development of accurate and efficient methods for analyzing multi-session, multi-task, and multi-subject neural data. It integrates six datasets from motor cortical regions, comprising `148 sessions` from `19 subjects` engaged in `7 distinct tasks`.
The figure here shows an illustration.

<img src='img/NeuroTask3.png' width='680px'>

## Recent updates:
- NeuroTask has been accepted as [Bernstein](https://www.world-wide.org/bernstein-24/neurotask-benchmark-dataset-multi-task-c8ba9ac5/) and [Cosyne - soon](https://www.world-wide.org/cosyne-25/conditional-diffusion-framework-31efbe03/) posters.
- NeuroTask nwb version was introduced at the [Open Neurodata Showcase '24](https://neurodatawithoutborders.github.io/nwb_hackathons/HCK20_2024_OpenNeurodataShowcase/)
- Dataset is now also available in NWB format on [Dandi](https://dandiarchive.org/dandiset/search?search=neurotask) and the [tutorial](tutorial_data_analysis_nwb.ipynb). 

## Dataset Download

**The public dataset suite is available for download through [Dandi](https://dandiarchive.org/dandiset/search?search=neurotask) in NWB format.** 

Each dandiset corresponds to a specific task.

**Alternatively, you can access the dataset in Parquet format through [Kaggle](https://www.kaggle.com/datasets/carolinafilipe/neurotask-multi-tasks-benchmark-dataset).** 

- **If you are using the web user interface**, you can download all data from the provided [link](https://www.kaggle.com/datasets/carolinafilipe/neurotask-multi-tasks-benchmark-dataset). The download button is at the upper right corner of the webpage.

- **If you would like to use the Kaggle API**, please follow the instructions [here](https://github.com/Kaggle/kaggle-api). After setting the API correctly, you can simply use the command below to download all data.

```
kaggle datasets download -d carolinafilipe/neurotask-multi-tasks-benchmark-dataset
```

## Using the NeuroTask Dataset

<img src='img/dataset.png' width='480px'>

The dataset schema is comprised of:

* Indexes for dataset,animal, session and trial
* Neurophysiological data.
* Behavior covariates.
* Events time indications.

Each row represents a single time step.

For more information on each of these components, please consult the provided [datasheet](docs/NeuroTask_datasheet.pdf).

## Loading the Dataset
We developed an API, `api_neurotask.py`, to streamline data loading and preprocessing tasks, such as data rebinning and alignment to specific start event with adjustable offsets. You can find the api tutorial here: [tutorial](tutorial_data_analysis_nwb.ipynb).

## Repository Structure

In this repository, you will find the code necessary to replicate the experiments of the presented paper.

The [first notebook](tutorial_data_analysis.ipynb) regards the loading, processing and visualization of the data (parquet format) and [nwb format](tutorial_data_analysis_nwb.ipynb) .


The [second notebook](tutorial_baselines.ipynb) presents the code to reproduce the benchmark results in the manuscript. 

## API <a name="API"></a>

`NeuroTask` has a minimalistic yet powerful API.

To load data in the NeuroTask schema format, you only need a single line of code.

Nwb version:
```python
from api_neurotask import *

data = nap.load_file("001056/sub-Animal-1-&-2/sub-Animal-1-&-2.nwb")

# Retrieve data filtered to include only rewarded trials.
df , bin = get_dataframe(data,filter_result=[b'R'])
```

Parquet version:
```python
from api_neurotask import *

parquet_file_path = 'NeuroTask/2_10_Chowdhury_CObump.parquet'

# Retrieve data filtered to include only rewarded trials.
df,bin = load_and_filter_parquet(parquet_file_path,['A', 'I','F'])
```

The following functions work the same for both versions. You can use the `rebin` and/or `align_event` functions as needed.

```python
# Rebin the dataset with a bin size of 20 ms 
df = rebin(df,prev_bin_size = bin ,new_bin_size = 20)

# Align each trial of the data (df) to a specific event ('EventTarget_Onset')
# The dataset has a bin size of 20 ms
#We want an offset of -20 ms before the event and 400 ms after the event

df = align_event(df, start_event='EventTarget_Onset', bin_size=20,offset_min=-20,offset_max=400)

```
Prepare the data for model fitting by processing it into input (X) and target (y) variables.

```python
# The input data (X) includes bins of spiking data from before the event, while the target (y) includes bins of future behavior variables (behavior_columns) to predict (e.g., 1 for the next time step prediction).
# The data is split into training, validation, and test sets according to the specified ranges.
# Each position in the resulting lists corresponds to a different session.

X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list = process_data(
    df, 
    bins_before=10, 
    bins_after=1,
    training_range=[0, 0.7], 
    valid_range=[0.7, 0.8], 
    testing_range=[0.8, 1], 
    behavior_columns=['hand_vel_x', 'hand_vel_y']
)
```

## Supported frameworks <a name="Supported-frameworks"></a>

NeuroTask nwb version works with ...

- [Pynapple](https://github.com/pynapple-org/pynapple)
- [Neurosift](https://neurosift.app/?p=/dandi)
- [NeMoS](https://github.com/flatironinstitute/nemos)

## License \& Acknowledgement
The NeuroTask benchmark dataset is released under a [CC BY-NC 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0). Our code implementation (api and baselines) is released under the [MIT License](https://opensource.org/licenses/MIT). We would also like to express our gratitude to the authors of baselines for releasing their code. And to the DANDI, NWB, and CCN software teams for their support and workshops.

## Contact Us
Please feel free to contact us if you have any questions about NeuroTask: carolina.filipe@research.fchampalimaud.org









