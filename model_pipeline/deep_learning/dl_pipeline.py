
from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset


class DeepLearningPipeline:
    def __init__(self,
                 dataset: Dataset,
                 tvt_ratio: dict[str, int],
                 model: nn,
                 training_options: dict[str, Any],
                 trained_model_save_path: str) -> None:

        self.dataset = dataset
        self.tvt_ratio = tvt_ratio
        self.model = model
        self.training_options = training_options
        self.trained_model_save_path = trained_model_save_path

    def _split_tvt(self) -> None:
        raise NotImplementedError

    def run_train(self) -> None:
        raise NotImplementedError

    def run_test(self) -> dict[Any]:
        raise NotImplementedError

    def run_inference(self, data_to_infer: Any) -> Any:
        return self.model(data_to_infer)

    def generate_test_report(self) -> str:
        raise NotImplementedError
