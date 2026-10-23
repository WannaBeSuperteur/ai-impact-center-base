
import os
import copy
from pathlib import Path
from typing import Any, Type, Callable

import cv2
import pandas as pd
import timm
import torch
import torchvision.transforms as transforms

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split

from deep_learning.dl_pipeline import DeepLearningPipeline
from deep_learning.common import SEED, seed_everything
from tqdm import tqdm


def category_by_dir_name(dataset_img_path: str) -> str:
    """Define (annotate) category using image path. (Category = directory name)"""

    return Path(dataset_img_path).parent.name


def create_dataset_df(train_data_path: str, category_func: Callable = category_by_dir_name) -> pd.DataFrame:
    """Create DataFrame for dataset, from given train_data_path."""

    base_path = Path(train_data_path).resolve()
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    absolute_img_paths = [
        str(p) for p in base_path.rglob("*")
        if p.suffix.lower() in valid_extensions
    ]
    categories = [
        category_func(img_path)
        for img_path
        in absolute_img_paths
    ]

    dataset_dict = {'category': categories, 'img_path': absolute_img_paths}
    dataset_df = pd.DataFrame(dataset_dict)
    return dataset_df


class VisionClsPipeline(DeepLearningPipeline):
    def __init__(self,
                 dataset: Dataset,
                 tvt_ratio: dict[str, float],
                 model: nn,
                 training_options: dict[str, Any],
                 trained_model_save_path: str,
                 transform: transforms.Compose) -> None:

        if not trained_model_save_path.endswith('.pth') and not trained_model_save_path.endswith('.pt'):
            raise ValueError(
                "trained_model_save_path must include full model file path, so must be ended with .pth or .pt")

        super().__init__(dataset, tvt_ratio, model, training_options, trained_model_save_path)
        self.transform = transform
        self.tvt_dataloader = {}
        self.dataset_size = len(dataset)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._init_val_and_test_result_dict()
        self.trained_model = None

    def _init_val_and_test_result_dict(self):
        self.val_result_dict = {'epoch': [], 'val_acc': [], 'val_loss': [], 'torch_memory': []}
        self.test_result = {}

    def _split_tvt(self) -> tuple[Any]:
        """Split train, valid, test dataset."""

        tvt_split_by_dir = self.training_options['tvt_split_by_dir']

        if tvt_split_by_dir:
            train_dataset = []
            valid_dataset = []
            test_dataset = []

            for data in self.dataset:
                path = data['img_path']
                if '/train/' in path:
                    train_dataset.append(data)
                elif '/valid/' in path:
                    valid_dataset.append(data)
                elif '/test/' in path:
                    test_dataset.append(data)

        else:
            train_dataset, valid_dataset, test_dataset = random_split(self.dataset,
                                                                      [self.train_size,
                                                                       self.valid_size,
                                                                       self.test_size])

        return train_dataset, valid_dataset, test_dataset

    def _create_dataloader_of_tvt(self) -> None:
        """Create DataLoader for train, valid, test dataset."""

        self.train_size = int(self.dataset_size * self.tvt_ratio['train'])
        self.valid_size = int(self.dataset_size * self.tvt_ratio['valid'])
        self.test_size = self.dataset_size - (self.train_size + self.valid_size)

        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_dataset, valid_dataset, test_dataset = self._split_tvt()

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

    def _run_train_one_epoch(self) -> dict:
        """Run vision classification model training. (one epoch)"""

        self.model.train()
        total = 0
        train_loss_sum = 0.0
        unsqueeze_label = self.training_options['unsqueeze_label']

        pbar = tqdm(enumerate(self.tvt_dataloader['train']), total=len(self.tvt_dataloader['train']))
        for idx, data in pbar:
            images, labels = data['img'], data['label']
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

    def _run_valid_or_test_one_epoch(self,
                                     is_test: bool = False,
                                     model_to_use: nn.Module | None = None) -> dict:

        """Run vision classification model validation/test. (one epoch)"""

        tvt_name = 'test' if is_test else 'valid'
        unsqueeze_label = self.training_options['unsqueeze_label']
        loader = self.tvt_dataloader[tvt_name]
        model = model_to_use if model_to_use is not None else self.model

        model.eval()
        correct, total = 0, 0
        val_loss_sum = 0
        val_result_detail_dict = {'pred_label': [], 'true_label': []}

        with torch.no_grad():
            for idx, data in enumerate(loader):
                images, labels = data['img'], data['label']
                images, labels = images.to(self.device), labels.to(self.device).to(torch.float32)
                outputs = model(images).to(torch.float32)

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
                    val_result_detail_dict['pred_label'].append(int(pred))
                    val_result_detail_dict['true_label'].append(int(label))

            val_accuracy = correct / total
            val_loss = val_loss_sum / total

        y_pred, y_true = val_result_detail_dict['pred_label'], val_result_detail_dict['true_label']
        cls_report = classification_report(y_true, y_pred)
        cf_matrix = confusion_matrix(y_true, y_pred)

        val_result_detail_df = pd.DataFrame(val_result_detail_dict)
        val_result_detail_df.to_csv(self.training_options['log_path'])

        return {'cls_report': cls_report,
                'cf_matrix': cf_matrix,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'val_result_detail_dict': val_result_detail_dict}

    def run_entire_training(self) -> nn.Module:
        self._create_dataloader_of_tvt()
        self._init_val_and_test_result_dict()

        current_epoch = 0
        min_val_loss_epoch = -1  # Loss-based Early Stopping
        min_val_loss = None
        best_epoch_model = None
        best_epoch_model_val_accuracy = None

        max_epochs = self.training_options['max_epochs']
        early_stopping_patience = self.training_options['early_stopping_patience']

        while True:
            self._run_train_one_epoch()
            val_result = self._run_valid_or_test_one_epoch()
            val_accuracy, val_loss = val_result['val_accuracy'], val_result['val_loss']
            val_loss = float(val_loss.detach().cpu())

            print(f'epoch={current_epoch}, val_acc={val_accuracy:.6f}, val_loss={val_loss:.6f}')

            self.val_result_dict['epoch'].append(current_epoch)
            self.val_result_dict['val_acc'].append(round(val_accuracy, 4))
            self.val_result_dict['val_loss'].append(round(val_loss, 4))
            self.val_result_dict['torch_memory'].append(torch.cuda.memory_allocated() // 1024)
            pd.DataFrame(self.val_result_dict).to_csv(self.training_options['log_path'])

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
        checked_val_result = self._run_valid_or_test_one_epoch(model_to_use=best_epoch_model)
        checked_val_accuracy, checked_val_loss = checked_val_result['val_accuracy'], checked_val_result['val_loss']

        print(f'[best model] val_acc={best_epoch_model_val_accuracy}, val_loss={min_val_loss}')
        print(f'[check] val_acc={checked_val_accuracy}, val_loss={checked_val_loss}')

        assert abs(best_epoch_model_val_accuracy - checked_val_accuracy) <= 1e-6
        assert abs(min_val_loss - checked_val_loss) <= 1e-6

        # run test
        print('testing ...')

        test_result = self._run_valid_or_test_one_epoch(is_test=True)
        test_accuracy, test_loss = test_result['val_accuracy'], test_result['val_loss']
        test_loss = float(test_loss.detach().cpu())

        self.val_result_dict['epoch'].append('test')
        self.val_result_dict['val_acc'].append(round(test_accuracy, 4))
        self.val_result_dict['val_loss'].append(round(test_loss, 4))
        self.val_result_dict['torch_memory'].append(torch.cuda.memory_allocated() // 1024)
        pd.DataFrame(self.val_result_dict).to_csv(self.training_options['log_path'])

        print(test_result)
        self.trained_model = best_epoch_model

    def run_test(self) -> dict:
        return self._run_valid_or_test_one_epoch(is_test=True)

    def run_inference(self, data_to_infer: Any) -> Any:
        return super().run_inference(data_to_infer)

    def generate_test_report(self) -> str:
        pass  # TODO implement

    def save_model(self):
        trained_model_dir = Path(self.trained_model_save_path).parent.name
        os.makedirs(trained_model_dir, exist_ok=True)

        if self.trained_model is not None:
            torch.save(self.trained_model.state_dict(), self.trained_model_save_path)


class VisionClsRunner:
    def __init__(self,
                 train_data_path: str,
                 dataset_class,
                 tvt_ratio: dict[str, float],
                 model_class: Type[nn],
                 model_name_or_path: str,
                 training_options: dict[str, Any],
                 trained_model_save_path: str,
                 transform: transforms.Compose,
                 load_model_weights_only: bool = True,
                 model_class_kwargs: dict | None = None):

        seed_everything()

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

    def _load_model(self, device, weights_only: bool = True, **kwargs):
        """Load vision model from model name (using timm) or path."""

        try:
            model = self.model_class(**kwargs)
            model.load_state_dict(torch.load(self.model_name_or_path,
                                             map_location=device,
                                             weights_only=weights_only))
        except:
            img_height, img_width = self.training_options['img_height'], self.training_options['img_width']

            if img_height is not None and img_width is not None:
                model = timm.create_model(self.model_name_or_path,
                                          in_chans=self.training_options['in_chans'],
                                          img_size=(img_height, img_width),
                                          num_classes=self.training_options['num_classes'],
                                          pretrained=True)
            else:
                model = timm.create_model(self.model_name_or_path,
                                          in_chans=self.training_options['in_chans'],
                                          num_classes=self.training_options['num_classes'],
                                          pretrained=True)

        model.optimizer = torch.optim.AdamW(model.parameters(), lr=self.training_options['lr'])

        scheduler_type = self.training_options['scheduler_type']
        if scheduler_type == 'exp':
            model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model.optimizer,
                                                                     gamma=self.training_options['gamma'])
        elif scheduler_type == 'lambda':
            model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model.optimizer,
                                                                         T_max=self.training_options['t_max'],
                                                                         eta_min=0)

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

    def run_all(self):
        self.run_model_train()
        self.run_model_test()
        self.generate_model_test_report()
        self.save_model()

