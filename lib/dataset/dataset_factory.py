from .datasets.pentathlon_dataset import PentathlonDataset

dataset_factory = {
    'MSRVTT': PentathlonDataset,
    'DiDeMo': PentathlonDataset,
    'YouCook2': PentathlonDataset,
}

def get_dataset(name):
    return dataset_factory[name]
