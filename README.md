# Custom Dataset Loader

This repository contains a Python class `NewDataset` designed for loading and preprocessing custom datasets for machine learning tasks. The class allows for easy integration of multiple datasets with specified input and target columns.

## Installation

To use this code, you need to install the required packages using pip:

```bash
pip install -q datasets==2.18.0
```

## Usage

### Class `NewDataset`

The `NewDataset` class is used for loading and preprocessing custom datasets. Here's how you can use it:

```python
from datasets import load_dataset, concatenate_datasets
from custom_dataset_loader import NewDataset

# Define datasets to load
datasets_to_load = {
    'path_to_dataset1': ('input_column_name1', 'target_column_name1'),
    'path_to_dataset2': ('input_column_name2', 'target_column_name2'),
    # Add more datasets as needed
}

# Initialize NewDataset object
custom_dataset = NewDataset(datasets_to_load)

# Optionally, apply mapping functions or perform data augmentation
# custom_dataset.map(fn, add_new=False, shuffle=False, **map_kwargs)

# Access dataset splits
splits = custom_dataset.splits
train_set = splits[0]
val_set = splits[1]
test_set = splits[2]

# Print dataset information
print(custom_dataset)
```

### Parameters

- `datasets`: A dictionary containing paths or presets of datasets along with input and target column names.
- `input_col_name`: Name of the input column in the datasets.
- `target_col_name`: Name of the target column in the datasets.

### Methods

- `map(fn, add_new=False, shuffle=False, **map_kwargs)`: Apply a mapping function to the dataset. Optionally add new data or shuffle the dataset.
- `splits`: Access the dataset splits (train, validation, test).
- `__str__()`: String representation of the dataset.

## Example

```python
# Example usage of NewDataset class

# Define datasets to load
datasets_to_load = {
    'path/to/train_dataset': ('input', 'target'),
    'path/to/val_dataset': ('input', 'target'),
    'path/to/test_dataset': ('input', 'target'),
}

# Initialize NewDataset object
custom_dataset = NewDataset(datasets_to_load)

# Apply mapping function (if needed)
# custom_dataset.map(fn, add_new=False, shuffle=False, **map_kwargs)

# Access dataset splits
splits = custom_dataset.splits
train_set = splits[0]
val_set = splits[1]
test_set = splits[2]

# Print dataset information
print(custom_dataset)
```
