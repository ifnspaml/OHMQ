from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
import abc
import math
import logging
import numpy as np
from pathlib import Path

import PIL

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data.dataset import T_co
from torchvision import transforms

DIR = os.path.abspath(os.path.dirname(__file__))
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class BaseDataset(Dataset, abc.ABC):
    """Base Class for datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], mode='train',
                 logger=logging.getLogger(__name__),
                 **kwargs):
        self.root = root

        try:
            self.train_data = os.path.join(root, self.files["train"])
            self.test_data = os.path.join(root, self.files["test"])
            self.val_data = os.path.join(root, self.files["val"])
        except AttributeError:
            pass

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            raise ValueError(
                'Files not found in specified directory: {}'.format(root))

    def __len__(self):
        return len(self.imgs)

    def __ndim__(self):
        return tuple(self.imgs.size())

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass


def exception_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class OpenImages(BaseDataset):
    """OpenImages dataset from [1].

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] https://storage.googleapis.com/openimages/web/factsfigures.html

    """
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, root=os.path.join(DIR, 'data/openimages'), mode='train',
                 crop_size=256,
        normalize=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        if mode == 'train':
            data_dir = self.train_data
        elif mode == 'validation':
            data_dir = self.val_data
        else:
            raise ValueError('Unknown mode!')

        path = Path(data_dir)
        self.imgs = list(path.glob('*.png')) + list(path.glob('*.jpg'))
        self.crop_size = crop_size
        self.image_dims = (3, self.crop_size, self.crop_size)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = normalize

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [# transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize((math.ceil(scale * H),
                                              math.ceil(scale * W))),
                           transforms.RandomCrop(self.crop_size),
                           transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        filesize = os.path.getsize(img_path)
        try:
            # This is faster but less convenient
            # H X W X C `ndarray`
            # img = imread(img_path)
            # img_dims = img.shape
            # H, W = img_dims[0], img_dims[1]
            # PIL
            img = PIL.Image.open(img_path)
            img = img.convert('RGB')
            W, H = img.size  # slightly confusing
            bpp = filesize * 8. / (H * W)

            shortest_side_length = min(H, W)

            minimum_scale_factor = \
                float(self.crop_size) / float(shortest_side_length)
            scale_low = max(minimum_scale_factor, self.scale_min)
            scale_high = max(scale_low, self.scale_max)
            scale = np.random.uniform(scale_low, scale_high)

            dynamic_transform = self._transforms(scale, H, W)
            transformed = dynamic_transform(img)
        except:
            print(f"Image {img_path} could not be loaded")
            return None

        # apply random scaling + crop, put each pixel
        # in [0.,1.] and reshape to (C x H x W)
        return transformed, bpp


class FolderDataset(Dataset):
    """
    Dataset class for inference on arbitrary image files.
    """
    def __init__(self, folder_path, transform_list):
        super().__init__()
        path = Path(folder_path)
        self.imgs = list(path.glob('**/*.png')) + list(path.glob('**/*.jpg'))
        self.transform_list = transform_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index) -> T_co:
        try:
            img = PIL.Image.open(self.imgs[index])
            img = img.convert('RGB')
            transformed = self.transform_list(img)
        except:
            print(f"Image {self.imgs[index]} could not be loaded")
            return None

        return transformed, 0


class SingleTensorDataset(Dataset):
    """
    A single Pytorch Tensor representing a dataset.
    It can be moved to GPU and does not perform any transformations to speed up dataloading quite a bit
    """
    def __init__(self, ds: Dataset):
        self.ds = ds
        tensors = [x[0] for x in ds]
        labels = [torch.tensor(x[1]) for x in ds]
        self.single_tensor = torch.stack(tensors)
        self.single_tensor_labels = torch.stack(labels)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.single_tensor[index], self.single_tensor_labels[index]

    def to(self, device):
        self.single_tensor = self.single_tensor.to(device)
        self.single_tensor_labels = self.single_tensor_labels.to(device)
        return self


class SingleTensorDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, drop_last=False):
        super().__init__(dataset, drop_last=drop_last, auto_collation=None, collate_fn=None)

    def fetch(self, idxs):
        return self.dataset.single_tensor[idxs], self.dataset.single_tensor_labels[idxs]


class SingleTensorLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False, device=None, drop_last=False):
        single_tensor_ds = SingleTensorDataset(dataset)
        super().__init__(single_tensor_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        self._dataset_fetcher = SingleTensorDatasetFetcher(self.dataset, drop_last=drop_last)
        if device:
            self.to(device)

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        iter_ = _SingleProcessDataLoaderIter(self)
        iter_._dataset_fetcher = SingleTensorDatasetFetcher(self.dataset, self.drop_last)
        return iter_

    def to(self, device):
        self.dataset.to(device)
        return self


def prepare_dataloaders(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not args.mnist:
        train_dataset = OpenImages(root=os.path.join(args.data_base_path, 'openimages'), logger=None, mode='train', normalize=True, crop_size=args.crop_size)
        val_dataset = OpenImages(root=os.path.join(args.data_base_path, 'openimages'), logger=None, mode='validation', normalize=True, crop_size=args.crop_size)
    else:
        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        dataset = MNIST('./data', transform=img_transform, download=True)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[int(len(dataset)*0.9), int(len(dataset)*0.1)])

    dataloaders = []
    dl_args = dict(batch_size=args.batch_size, drop_last=True)
    for ds, is_train in [(train_dataset, True), (val_dataset, False)]:
        dl_args['shuffle'] = is_train
        n_samples = args.num_train_samples if is_train else args.num_val_samples
        ds = ds if n_samples == -1 else torch.utils.data.Subset(ds, range(min(len(ds), n_samples)))
        if args.mnist and torch.cuda.is_available():
            dl = SingleTensorLoader(ds, device=device, **dl_args)
        else:
            dl = DataLoader(ds, num_workers=args.num_workers, collate_fn=exception_collate_fn, **dl_args)
        mode = 'training' if is_train else 'val'
        setattr(args, f'num_{mode}_batches', len(dl))
        print(f'Number of {mode} images: {len(ds)}')
        print(f'Number of {mode} batches: {len(dl)}')
        dataloaders.append(dl)
    return dataloaders
