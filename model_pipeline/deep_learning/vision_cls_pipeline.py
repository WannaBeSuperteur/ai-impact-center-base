
import copy
from pathlib import Path
from typing import Any, Type, Callable

import cv2
import device
import pandas as pd
import timm
import torch
import torchvision.transforms as transforms

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split

from dl_pipeline import DeepLearningPipeline
from common import SEED


def category_by_dir_name(dataset_img_path: str) -> str:
    """Define (annotate) category using image path. (Category = directory name)"""

    return Path(dataset_img_path).parent.name


def create_dataset_df(train_data_path: str, category_func: Callable = category_by_dir_name) -> pd.DataFrame:
    """Create DataFrame for dataset, from given train_data_path."""

    relative_img_paths = [
        str(p.relative_to(train_data_path))
        for p in train_data_path.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
    categories = [
        category_func(img_path)
        for img_path
        in relative_img_paths
    ]

    dataset_dict = {'category': categories, 'img_path': relative_img_paths}
    dataset_df = pd.DataFrame(dataset_dict)
    return dataset_df


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
        self.tvt_dataloader = {}
        self.dataset_size = len(dataset)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _split_tvt(self) -> None:
        """Split train, valid, test dataset."""

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

    def _run_train_one_epoch(self, unsqueeze_label: bool = False) -> dict:
        """Run vision classification model training. (one epoch)"""

        self.model.train()
        total = 0
        train_loss_sum = 0.0

        for idx, (images, labels) in enumerate(self.tvt_dataloader['train']):
            images, labels = images.to(self.device), labels.to(self.device).to(torch.float32)

            self.model.optimizer.zero_grad()
            outputs = self.model(images).to(torch.float32)

            if unsqueeze_label:
                loss = self.loss_func(outputs, labels.unsqueeze(1))
            else:
                loss = self.loss_func(outputs, labels)
            loss.backward()
            self.model.optimizer.step()

            train_loss_sum += loss.item()
            total += labels.size(0)

        train_loss = train_loss_sum / total
        return {'train_loss': train_loss}

    def _run_valid_or_test_one_epoch(self, is_test: bool = False, unsqueeze_label: bool = False) -> dict:
        """Run vision classification model validation/test. (one epoch)"""

        tvt_name = 'test' if is_test else 'valid'
        loader = self.tvt_dataloader[tvt_name]

        self.model.eval()
        correct, total = 0, 0
        val_loss_sum = 0
        val_result_detail_dict = {'pred_label': [], 'true_label': []}

        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device).to(torch.float32)
                outputs = self.model(images).to(torch.float32)

                if unsqueeze_label:
                    val_loss_batch = self.loss_func(outputs, labels.unsqueeze(1))
                else:
                    val_loss_batch = self.loss_func(outputs, labels)
                val_loss_sum += val_loss_batch

                _, predicted = torch.max(outputs, 1)
                if not unsqueeze_label:
                    _, labels = torch.max(labels, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predicted_ = list(predicted.detach().cpu().numpy())
                labels_ = list(labels.detach().cpu().numpy())

                for pred, label in zip(predicted_, labels_):
                    val_result_detail_dict['pred_label'].append(pred)
                    val_result_detail_dict['true_label'].append(label)

            val_accuracy = correct / total
            val_loss = val_loss_sum / total

        y_pred, y_true = val_result_detail_dict['pred_label'], val_result_detail_dict['true_label']
        cls_report = classification_report(y_true, y_pred)
        cf_matrix = confusion_matrix(y_true, y_pred)

        return {'cls_report': cls_report,
                'cf_matrix': cf_matrix,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'val_result_detail_dict': val_result_detail_dict}

    def run_entire_training(self, unsqueeze_label: bool = False) -> nn.Module:
        self._split_tvt()

        current_epoch = 0
        min_val_loss_epoch = -1  # Loss-based Early Stopping
        min_val_loss = None
        best_epoch_model = None
        best_epoch_model_val_accuracy = None
        val_loss_list = []

        max_epochs = self.training_options['max_epochs']
        early_stopping_patience = self.training_options['early_stopping_patience']

        while True:
            self._run_train_one_epoch(unsqueeze_label=unsqueeze_label)
            val_result = self._run_valid_or_test_one_epoch(unsqueeze_label=unsqueeze_label)
            val_accuracy, val_loss = val_result['val_accuracy'], val_result['val_loss']

            print(f'epoch={current_epoch}, val_acc={val_accuracy:.6f}, val_loss={val_loss:.6f}')
            val_loss_list.append(val_loss)

            # update scheduler
            if self.model.scheduler is not None:
                self.model.scheduler.step()

            # update best epoch model
            if min_val_loss is None or val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_epoch = current_epoch
                best_epoch_model_val_accuracy = val_accuracy

                pretrained_model = copy.deepcopy(self.model)
                best_epoch_model = pretrained_model.to(self.device)
                best_epoch_model.device = self.device
                best_epoch_model.load_state_dict(self.model.state_dict())

            if current_epoch + 1 >= max_epochs or current_epoch - min_val_loss_epoch >= early_stopping_patience:
                break

            current_epoch += 1

        # assert best epoch model accuracy & loss
        checked_val_result = self._run_valid_or_test_one_epoch(unsqueeze_label=unsqueeze_label)
        checked_val_accuracy, checked_val_loss = checked_val_result['val_accuracy'], checked_val_result['val_loss']

        print(f'[best model] val_acc={best_epoch_model_val_accuracy}, val_loss={min_val_loss}')
        print(f'[check] val_acc={checked_val_accuracy}, val_loss={checked_val_loss}')

        assert abs(best_epoch_model_val_accuracy - checked_val_accuracy) <= 1e-6
        assert abs(min_val_loss - checked_val_loss) <= 1e-6

        # run test
        print('testing ...')
        test_result = self._run_valid_or_test_one_epoch(is_test=True, unsqueeze_label=unsqueeze_label)

        # TODO valid, test result logging
        return best_epoch_model

    def run_test(self) -> dict:
        return self._run_valid_or_test_one_epoch(is_test=True)

    def run_inference(self, data_to_infer: Any) -> Any:
        return super().run_inference(data_to_infer)

    def generate_test_report(self) -> str:
        pass  # TODO implement

    def save_model(self):
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

        self.model = self._load_model(device=device,
                                      weights_only=load_model_weights_only,
                                      **model_class_kwargs)
        self.dataset = self._load_dataset()
        self.pipeline = VisionClsPipeline(dataset=self.dataset,
                                          tvt_ratio=self.tvt_ratio,
                                          model=self.model,
                                          training_options=self.training_options,
                                          trained_model_save_path=self.trained_model_save_path,
                                          transform=self.transform)

    def _load_model(self, device: device, weights_only: bool = True, **kwargs):
        """Load vision model from model name (using timm) or path."""

        try:
            model = self.model_class(**kwargs)
            model.load_state_dict(torch.load(self.model_name_or_path,
                                             map_location=device,
                                             weights_only=weights_only))
        except:
            model = timm.create_model(self.model_name_or_path,
                                      pretrained=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        return model

    def _load_dataset(self) -> Dataset:
        dataset_df = create_dataset_df(self.train_data_path)
        return self.dataset_class(dataset_df, self.transform)

    def run_model_train(self):
        self.pipeline.run_entire_training()

    def run_model_test(self):
        self.pipeline.run_test()

    def run_model_inference(self, data_to_infer: Any):
        return self.pipeline.run_inference(data_to_infer)

    def run_model_inference_from_img_path(self, img_path: Any, flags: int = cv2.IMREAD_COLOR):
        img = cv2.imread(img_path, flags=flags)
        img = self.transform(img)
        return self.run_model_inference(img)

    def generate_model_test_report(self):
        self.model.generate_test_report()

    def save_model(self):
        self.model.save_model()

