# required
# pip install -q datasets==2.18.0

from datasets import load_dataset, concatenate_datasets

class NewDataset():
    def __init__(self, datasets, dataset_preprocesses = {}, input_col_name='inp', target_col_name='target'):
        """
          Assuming that 'datasets' is look like:

            {
              <dataset path or preset>: (<input col name>, <target col name>),
              ...
            }

          If u also want to preprocess datasets at first you can u use 'dataset_preprocesses': 
          Assuming that 'dataset_preprocesses' is look like: 

            {
              <dataset path or preset>: <fn>
            }
        """
        self.inp=input_col_name
        self.target=target_col_name

        self.dict_dataset = {
            'train': [],
            'validation': [],
            'test': []
        }

        for (name, (inp, target)) in datasets.items():
          dataset = load_dataset(path = name) # Load Dataset from HUgging Face

          if name in dataset_preprocesses.keys(): 
            print(f'{name} have a custom fn')
            dataset = dataset_preprocesses[name](dataset)

          dataset = dataset.select_columns([inp, target]) # remove useless columns
            # prepare cols names
          if inp != self.inp:
            dataset = dataset.rename_column(inp, self.inp)
          if target != self.target:
            dataset = dataset.rename_column(target, self.target)

          assert 'train' in list(dataset.keys())

          for k in self.dict_dataset.keys():
            self.dict_dataset[k].append(dataset[k])

        self.dict_dataset = {k: concatenate_datasets(v) for k, v in self.dict_dataset.items()}

    def map(self, fn, add_new=False, shuffle=False, **map_kwargs):
      for split, dataset in self.dict_dataset.items():

        if add_new == True:
          new_dataset = dataset.map(fn, **map_kwargs)
          self.dict_dataset[split] = concatenate_datasets([self.dict_dataset[split], new_dataset])

        else:
          self.dict_dataset[split] = dataset.map(fn, **map_kwargs)

        if shuffle == True:
          self.dict_dataset[split].shuffle()
      return self

    @property
    def splits(self):
      return [v for k, v in self.dict_dataset.items()]

    def __str__(self) -> str:
        return str(self.dict_dataset)
