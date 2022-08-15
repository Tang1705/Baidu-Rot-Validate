from torchvision.transforms import Compose
from torchvision.transforms import transforms

from Dataset import DatasetFromFolder, DatasetFromTestFolder


def input_transform():
    return Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def get_training_set(data_dir):
    return DatasetFromFolder(data_dir,
                             transform=input_transform())

def get_testing_set(data_dir):
    return DatasetFromTestFolder(data_dir,
                             transform=input_transform())
