import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

"""
template = ['a photo of a {}.']
"""
template = ['a photo of a pig {}.']



class EuroSAT(DatasetBase):

    dataset_dir = 'pigs'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'cropped_images')
        self.split_path = os.path.join(self.dataset_dir, 'dataset_crops.json')
        
        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    