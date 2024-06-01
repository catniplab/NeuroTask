# The NeuroTask Benchmark Dataset

NeuroTask is a benchmark dataset designed to facilitate the development of accurate and efficient methods for analyzing multi-session, multi-task, and multi-subject neural data. It integrates six datasets from motor cortical regions, comprising `148 sessions` from `17 subjects` engaged in `7 distinct tasks`.
The figure here shows an illustration.

<img src='img/NeuroTask3.png' width='780px'>

| ID | Task                   | #Subj. | #Sess. | #Neurons | #Trials | Dataset            | Brain Area |
|----|------------------------|--------|--------|----------|---------|--------------------|--------------------|
| 1  | Random Target          | 2      | 47     | 18406    | 25483   | Makin et al.       | M1, S1 |
| 2  | Center-Out with Bump  | 2      | 4      | 461      | 2766    | Chowdhury et al.   | Area 2 |
| 2  | Two-Workspace          | 3      | 9      | 629      | 4515    |                    |
| 3  | Center-Out             | 4      | 30     | 1827     | 9226    | Gallego et al.     |M1, Area 2, PMd |
| 4  | Center-Out             | 2      | 23     | 2194     | 4712    |                    |
| 4  | Wrist Isometric Center-Out | 1  | 13     | 899      | 2766    | Ma Xuan et al.     |M1 |
| 4  | Key Grasp              | 1      | 9      | 864      | 903     |                    |
| 5  | Center-Out             | 2      | 4      | 681      | 763     | Dyer et al.        |M1 |
| 6  | Maze                   | 2      | 9      | 1728     | 23117   | Churchland et al.  |M1, PMd |


## Datasets Download

**The public dataset suite is available for download through [Kaggle](https://www.kaggle.com/datasets/carolinafilipe/neurotask-multi-tasks-benchmark-dataset).** 

- **If you are using the web user interface**, you can download all data from the provided [link](https://www.kaggle.com/datasets/carolinafilipe/neurotask-multi-tasks-benchmark-dataset). The download button is at the upper right corner of the webpage.

- **If you would like to use the Kaggle API**, please follow the instructions [here](https://github.com/Kaggle/kaggle-api). After setting the API correctly, you can simply use the command below to download all data.

```
kaggle datasets download -d carolinafilipe/neurotask-multi-tasks-benchmark-dataset
```

## Loading the Dataset
We developed an API, `api_neurotask.py`, to streamline data loading and preprocessing tasks, such as data rebinning and alignment to specific start event with adjustable offsets. You can find the api tutorial here: [tutorial](tutorial_data_analysis.ipynb).

## Repository Structure

In this repository, you will find the code necessary to replicate the experiments of the presented paper.

The [first notebook](tutorial_data_analysis.ipynb) regards the loading, processing and visualization of the data.

The [second notebook](tutorial_baselines.ipynb) presents the code to reproduce the benchmark results in the manuscript. 





