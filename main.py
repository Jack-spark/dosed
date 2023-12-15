import os
import json

h5_directory = './data/h5'  # adapt if you used a different DOWNLOAD_PATH when running `make download_example`
import torch
import tempfile
import json
import random
import sys
sys.path.append("..")

from dosed.utils import Compose
from dosed.datasets import BalancedEventDataset as dataset
from dosed.models import DOSED3 as model
from dosed.models import TC
from dosed.datasets import get_train_validation_test
from dosed.trainers import trainers
from dosed.preprocessing import GaussianNoise, RescaleNormal, Invert
from dosed.functions import augmentations
from dosed.functions import compute_metrics_dataset
from config_files import EDF_Configs

seed = 2019
configs = EDF_Configs.Config()
train, validation, test = get_train_validation_test(h5_directory,
                                                    percent_test=25,
                                                    percent_validation=33,
                                                    seed=seed)

print("Number of records train:", len(train))
print("Number of records validation:", len(validation))
print("Number of records test:", len(test))
train = train[0: 2]
window = 10  # window duration in seconds
ratio_positive = 0.5  # When creating the batch, sample containing at least one spindle will be drawn with that probability
# 一个训练批次中，每个样本包含至少一个脑电活动（spindle）的概率
fs = 32

signals = [
    {
        'h5_path': '/eeg_0',
        'fs': 64,
        'processing': {
            "type": "clip_and_normalize",
            "args": {
                    "min_value": -150,
                "max_value": 150,
            }
        }
    },
    {
        'h5_path': '/eeg_1',
        'fs': 64,
        'processing': {
            "type": "clip_and_normalize",
            "args": {
                    "min_value": -150,
                "max_value": 150,
            }
        }
    }
]

events = [
    {
        "name": "spindle",
        "h5_path": "spindle",
    },
]
dataset_parameters = {
    "h5_directory": h5_directory,#h5目录
    "signals": signals,#信号，包括两个通道
    "events": events,#事件，只有一个事件，即脑电活动（spindle）
    "window": window,#窗口，10s
    "fs": fs,#采样率32
    "ratio_positive": ratio_positive,#一个训练批次中，每个样本包含至少一个脑电活动（spindle）的概率
    "n_jobs": -1,  # Make use of parallel computing to extract and normalize signals from h5
    "cache_data": True,  # by default will store normalized signals extracted from h5 in h5_directory + "/.cache" directory
}

dataset_validation = dataset(records=validation, **dataset_parameters)
dataset_test = dataset(records=test, **dataset_parameters)

# for training add data augmentation
dataset_parameters_train = {
    "transformations": Compose([
        GaussianNoise(),
        RescaleNormal(),
        Invert(),
    ])
}
dataset_parameters_train.update(dataset_parameters)#更新dataset_parameters的键值对到dataset_parameters_train
dataset_train = dataset(records=train, **dataset_parameters_train)# inputsize=2,320
default_event_sizes = [0.7, 1, 1.3]#如何选取默认事件的大小，要检测的事件是脑电活动（spindle），大小约为1s
k_max = 5
kernel_size = 5
probability_dropout = 0.1
device = torch.device("cuda")
sampling_frequency = dataset_train.fs

net_parameters = {
    "detection_parameters": {
        "overlap_non_maximum_suppression": 0.5,
        "classification_threshold": 0.7
    },
    "default_event_sizes": [
        default_event_size * sampling_frequency
        for default_event_size in default_event_sizes
    ],
    "k_max": k_max,
    "kernel_size": kernel_size,
    "pdrop": probability_dropout,
    "fs": sampling_frequency,   # just used to print architecture info with right time
    "input_shape": dataset_train.input_shape,# 2，10s，采样率32hz
    "number_of_classes": dataset_train.number_of_classes,# 1
}
net = model(**net_parameters)
net = net.to(device)

temporal_contr_model = TC(configs, device).to(device)

optimizer_parameters = {
    "lr": 5e-3,
    "weight_decay": 1e-8,
}
loss_specs = {
    "type": "focal",
    "parameters": {
        "number_of_classes": dataset_train.number_of_classes,
        "device": device,
    }
}

epochs = 1

trainer = trainers["adam"](
    net,
    optimizer_parameters=optimizer_parameters,
    loss_specs=loss_specs,
    epochs=epochs,
)
if __name__ == "__main__":
    best_net_2record = trainer.train(
            dataset_train,
            dataset_validation,
            drop_last=True,
        )
    torch.save(best_net_2record.state_dict(), "best_net_2record.pth")


# if __name__ == "__main__":
#     # best_net_train, best_metrics_train, best_threshold_train = trainer.train(
#     #     dataset_train,
#     #     dataset_validation,
#     #     training_mode="self_supervised",
#     #     temporal_contr_model=temporal_contr_model,
#     #     config=configs,
#     #     drop_last=True,
#     # )
#     # torch.save(best_net_train.state_dict(), "best_net_train_self_epoch=20(12.14).pth")
#     best_net_train = torch.load("best_net_train.pth")
#     best_net_train_self = torch.load("best_net_train(self)20.pth")
#     import pickle
#     with open('minimum_example/hyperparameters.pkl', 'rb') as f:
#         hyperparameters = pickle.load(f)
#     print(hyperparameters)
#     # best_metrics_test = compute_metrics_dataset(
#     #     best_net_train,
#     #     dataset_test,
#     #     threshold=hyperparameters['best_threshold_train'],
#     # )
#     # print(best_metrics_test)
#     # best_metrics_train_self = compute_metrics_dataset(
#     #     best_net_train_self,
#     #     dataset_train,
#     #     threshold=hyperparameters['best_threshold_train'],
#     # )
#     # print(best_metrics_train_self)
#     # best_metrics_test_self = compute_metrics_dataset(
#     #     best_net_train_self,
#     #     dataset_test,
#     #     threshold=hyperparameters['best_threshold_train'],
#     # )
#     # print(best_metrics_test_self)
#     net_parameter = torch.load('best_net_train_self_epoch=20(12.14).pth')
#     # 更新参数到net
#     net.load_state_dict(net_parameter)
#     best_metrics_train_self = compute_metrics_dataset(
#         net,
#         dataset_train,
#         threshold=hyperparameters['best_threshold_train'],
#     )
#     print(best_metrics_train_self)
#     best_metrics_test_self = compute_metrics_dataset(
#         net,
#         dataset_test,
#         threshold=hyperparameters['best_threshold_train'],
#     )
#     print(best_metrics_test_self)