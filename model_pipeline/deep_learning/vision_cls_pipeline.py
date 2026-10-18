
from pathlib import Path
from typing import Any, Type, Callable

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader, random_split

from dl_pipeline import DeepLearningPipeline
from common import SEED


def category_by_dir_name(dataset_img_path: str) -> str:
    """Define (annotate) category using image path. (Category = directory name)"""

    return Path(dataset_img_path).parent.name


def create_dataset_df(train_data_path: str, category_func: Callable = category_by_dir_name) -> pd.DataFrame:
    """Create DataFrame for dataset, from given train_data_path."""

    dataset_dict = {'category': [], 'img_path': []}
    pass  # TODO implement


class VisionClsPipeline(DeepLearningPipeline):
    def __init__(self,
                 dataset: Dataset,
                 tvt_ratio: dict[str, int],
                 model: nn,
                 training_options: dict[str, Any],
                 trained_model_save_path: str,
                 transform: transforms.Compose) -> None:

        super().__init__(dataset, tvt_ratio, model, training_options, trained_model_save_path)
        self.transform = transform
        self.tvt_dataloader = {'train': None, 'valid': None, 'test': None}
        self.dataset_size = len(dataset)

    def _split_tvt(self) -> None:
        train_size = int(self.dataset_size * self.tvt_ratio['train'])
        valid_size = int(self.dataset_size * self.tvt_ratio['valid'])
        test_size = self.dataset_size - (train_size + valid_size)

        generator = torch.Generator()
        generator.manual_seed(SEED)

        train_dataset, valid_dataset, test_dataset = random_split(self.dataset,
                                                                  [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.training_options['train_batch_size'],
                                  shuffle=True,
                                  generator=generator)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.training_options['valid_batch_size'],
                                  shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.training_options['test_batch_size'],
                                 shuffle=False)

        self.tvt_dataloader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    def run_train(self) -> None:
        self._split_tvt()
        pass  # TODO implement

    def run_test(self) -> dict[Any]:
        pass  # TODO implement

    def run_inference(self, data_to_infer: Any) -> Any:
        return super().run_inference(data_to_infer)

    def generate_test_report(self) -> str:
        pass  # TODO implement


class VisionClsRunner:
    def __init__(self,
                 train_data_path: str,
                 dataset_class: Type[Dataset],
                 tvt_ratio: dict[str, int],
                 model_class: Type[nn],
                 model_name_or_path: str,
                 training_options: dict[str, Any],
                 trained_model_save_path: str,
                 transform: transforms.Compose,
                 load_model_weights_only: bool = True,
                 model_class_kwargs: dict | None = None):

        self.train_data_path = train_data_path
        self.dataset_class = dataset_class
        self.tvt_ratio = tvt_ratio
        self.model_class = model_class
        self.model_name_or_path = model_name_or_path
        self.training_options = training_options
        self.trained_model_save_path = trained_model_save_path
        self.transform = transform

        if model_class_kwargs is None:
            model_class_kwargs = {}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._load_model(map_location=device,
                                      weights_only=load_model_weights_only,
                                      **model_class_kwargs)
        self.dataset = self._load_dataset()
        self.pipeline = VisionClsPipeline(dataset=self.dataset,
                                          tvt_ratio=self.tvt_ratio,
                                          model=self.model,
                                          training_options=self.training_options,
                                          trained_model_save_path=self.trained_model_save_path,
                                          transform=self.transform)

    def _load_model(self, map_location: Any, weights_only: bool = True, **kwargs):
        model = self.model_class(**kwargs)
        model.load_state_dict(torch.load(self.model_name_or_path,
                                         map_location=map_location,
                                         weights_only=weights_only))
        return model

    def _load_dataset(self) -> Dataset:
        dataset_df = create_dataset_df(self.train_data_path)
        return self.dataset_class(dataset_df, self.transform)

    def run_model_train(self):
        self.pipeline.run_train()

    def run_model_test(self):
        self.pipeline.run_test()

    def run_model_inference(self, data_to_infer: Any):
        return self.pipeline.run_inference(data_to_infer)

    def run_model_inference_from_img_path(self, img_path: Any, flags: int):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.transform(img)
        return self.run_model_inference(img)

    def generate_model_test_report(self):
        self.model.generate_test_report()
