import random
from typing import List
import torch

from fairseq.data import BaseWrapperDataset, data_utils

class RandomInputDataset(BaseWrapperDataset):
    def __init__(
        self, 
        dataset, # audio data
        text_dataset,
        input_key: str,
        add_to_input, #? 
        pad_idx, #?
    ):
        super().__init__(dataset)
        self.text_dataset = text_dataset
        # if isinstance(input_key_path, str):
        #     input_key_path = [input_key_path]
        assert input_key is str
        self.input_key = input_key
        self.add_to_input = add_to_input
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        item: dict = self.dataset[index]
        item[self.input_key] = random.choice(self.text_dataset)
        return item

    def collater(self, samples: List[dict]):
        """
        samples: [{}, {}, {}]
        {'id': 55, 'features': [1, 2, 3],'random_label': [A, B, C]}
        num 0 in features tensor means padding
        """
        if len(samples) == 0:
            return {}

        features = [ s['features'] for s in samples]
        sizes = [ len(feat) for feat in features]

        target_size = max(sizes)
        collated_features = features[0].new_zeros(
            len(features), target_size
        )

        collated_features[:features.shape[0], :features.shape[1]] = features

        indices = torch.tensor([s["id"] for s in samples], dtype=torch.int64)
        indices




        # collated = [for sample in samples]

