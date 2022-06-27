import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))


        ids_to_lbls = pd.DataFrame(image_id_to_label, index=[0]).T
        ids_to_lbls["label"] = ids_to_lbls[0]
        ids_to_lbls["index"] = ids_to_lbls.index
        del ids_to_lbls[0]

        X_val, X_test, Y_val, Y_test, X_train, Y_train = self.split_data(ids_to_lbls)


        image_ids, labels = None, None
        if self._split == 'train':
            image_ids, labels = X_train, Y_train
        elif self._split == 'val':
            image_ids, labels = X_val, Y_val
        elif self._split == 'test':
            image_ids, labels = X_test, Y_test
        image_ids["index"] = image_ids["index"].apply(
            lambda image_id: self._images_folder / f"image_{image_id:05d}.jpg")

        self._labels = labels.values
        self._image_files = image_ids['index'].values

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

    def split_data(self, ids_to_lbls):
        X_train, X_test, Y_train, Y_test = train_test_split(ids_to_lbls[["index"]], ids_to_lbls["label"], test_size=0.5)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
        return X_val, X_test, Y_val, Y_test, X_train, Y_train



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    ep, train_acc, train_loss, val_acc, val_loss = [], [], [], [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        ep.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_acc.append(epoch_acc.item())
                train_loss.append(epoch_loss)
            elif phase == 'valid':
                val_acc.append(epoch_acc.item())
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                time_elapsed = time.time() - since
                print('Time from Start {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #     train_df = pd.DataFrame({'epoch': ep, 'train_accuracy': train_acc, 'train_loss': train_loss, 'validation_accuracy': val_acc, 'validation_loss': val_loss})
    #     train_df.to_csv('train_results.csv', index=False)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [ep, train_acc, train_loss, val_acc, val_loss]


def evaluate_model(model):
    model.eval()
    res = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            res += torch.sum(preds.double() == labels.data)
    print('Evaluating the model on the test data set')
    print("Accuracy:", (res / dataset_sizes['test']).item())


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((230, 230)),
            transforms.RandomRotation(30, ),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
    }

    train_data = Flowers102(root="data", split="train", download=True, transform=data_transforms['train'])

    test_data = Flowers102(root="data", split="test", download=True, transform=data_transforms['valid'])

    valid_data = Flowers102(root="data", split="val", download=True, transform=data_transforms['valid'])

    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'valid', 'test']}

    ##VGG19
    vgg19_model = models.vgg19_bn(pretrained=True)
    num_ftrs = vgg19_model.classifier[0].in_features
    num_class = 102
    vgg19_model.fc = nn.Linear(num_ftrs, num_class)
    vgg19_model = vgg19_model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(vgg19_model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    vgg19_model, vgg_results = train_model(vgg19_model, criterion, optimizer_ft, exp_lr_scheduler,
                                           num_epochs=10)

    evaluate_model(vgg19_model)

    # Train acc
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(vgg_results[0], vgg_results[1])
    plt.show()
    plt.savefig('TrainAcc.png')

    # Train CrossEntropy
    plt.title('Train Loss(CrossEntropy)')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    plt.plot(vgg_results[0], vgg_results[2])
    plt.show()

    # val acc
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(vgg_results[0], vgg_results[3])
    plt.show()

    # val CrossEntropy
    plt.title('Validation Loss(CrossEntropy)')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    plt.plot(vgg_results[0], vgg_results[4])
    plt.show()

    ## RESNET 50

    resnet50_model = models.resnext50_32x4d(pretrained=True)
    num_ftrs = resnet50_model.fc.in_features
    resnet50_model.fc = nn.Linear(num_ftrs, num_class)

    resnet50_model = resnet50_model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(resnet50_model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    resnet50_model, resnet_results = train_model(resnet50_model, criterion, optimizer_ft, exp_lr_scheduler,
                                                 num_epochs=10)

    evaluate_model(resnet50_model)

    # Train acc
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(resnet_results[0], resnet_results[1])
    plt.show()

    # Train CrossEntropy
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    plt.plot(resnet_results[0], resnet_results[2])
    plt.show()

    # Val acc
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(resnet_results[0], resnet_results[3])
    plt.show()

    # val crossentropy
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    plt.plot(resnet_results[0], resnet_results[4])
    plt.show()

