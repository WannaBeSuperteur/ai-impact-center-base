
from typing import Any, Type

import cv2
import torch
import torchvision.transforms as transforms
from datasets import Dataset
from torch import nn

from dl_pipeline import DeepLearningPipeline


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

    def run_train(self) -> None:
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
                 train_data_ctg_option: str,
                 tvt_ratio: dict[str, int],
                 model_class: Type[nn],
                 model_name_or_path: str,
                 training_options: dict[str, Any],
                 trained_model_save_path: str,
                 transform: transforms.Compose,
                 load_model_weights_only: bool = True,
                 model_class_kwargs: dict | None = None):

        self.train_data_path = train_data_path
        self.train_data_ctg_option = train_data_ctg_option
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
        pass  # TODO implement

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
