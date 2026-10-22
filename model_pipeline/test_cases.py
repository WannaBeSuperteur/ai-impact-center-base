import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

from deep_learning.vision_cls_pipeline import VisionClsRunner


DEFAULT_TRAINING_OPTIONS = {'train_batch_size': 16,
                            'valid_batch_size': 4,
                            'test_batch_size': 4,
                            'max_epochs': 100,
                            'early_stopping_patience': 10,
                            'lr': 0.001,
                            'scheduler_type': 'exp',
                            'gamma': 0.95}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


class DefaultClassificationDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_path = dataset_df['img_path'].tolist()
        self.label = dataset_df['category'].tolist()
        self.transform = transform

        self.label_mapping = {}
        self.label_list = sorted(list(set(self.label)))
        for idx, label in enumerate(self.label_list):
            self.label_mapping[label] = idx

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = read_image(img_path)
        img = self.transform(img)

        label = self.label[idx]
        label_int = self.label_mapping[label]
        label_one_hot = np.eye(len(self.label_list))[label_int]

        return img, label_one_hot


def test_case_1():
    tc1_runner = VisionClsRunner(train_data_path='../test_cases/deep_learning_vision_cls_01/archive/mnist_sample',
                                 dataset_class=DefaultClassificationDataset,
                                 tvt_ratio={'train': 0.75, 'valid': 0.15, 'test': 0.1},
                                 model_class=None,
                                 model_name_or_path='resnet18',
                                 training_options=DEFAULT_TRAINING_OPTIONS,
                                 trained_model_save_path='../test_cases/deep_learning_vision_cls_01/trained_model/model.pth',
                                 transform=DEFAULT_TRANSFORM)
    tc1_runner.run_all(log_path='../test_cases/deep_learning_vision_cls_01/log.csv')


if __name__ == '__main__':
    test_case_1()
