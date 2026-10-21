
from deep_learning.vision_cls_pipeline import VisionClsRunner


DEFAULT_TRAINING_OPTIONS = {'train_batch_size': 16,
                            'valid_batch_size': 4,
                            'test_batch_size': 4,
                            'max_epochs': 100,
                            'early_stopping_patience': 10,
                            'lr': 0.001}


def test_case_1():
    tc1_runner = VisionClsRunner(train_data_path='../test_cases/deep_learning_vision_cls_01/archive/mnist_sample',
                                 dataset_class=None,
                                 tvt_ratio={'train': 0.75, 'valid': 0.15, 'test': 0.1},
                                 model_class=None,
                                 model_name_or_path='resnet18',
                                 training_options=DEFAULT_TRAINING_OPTIONS,
                                 trained_model_save_path='../test_cases/deep_learning_vision_cls_01/trained_model/model.pth',
                                 transform=None)
    tc1_runner.run_all(log_path='../test_cases/deep_learning_vision_cls_01/log.csv')


if __name__ == '__main__':
    test_case_1()
