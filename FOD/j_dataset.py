import os

# pyright: reportGeneralTypeIssues=false
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from FOD.j_utils import get_total_paths, get_splitted_dataset, get_transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


def gen_dataloader_from_list_data(gds, type_str, num_samples):
    autofocus_datasets = [AutoFocusDataset(gd, type_str, num_samples) for gd in gds]
    b_dones = [d.b_done for d in autofocus_datasets]
    data = ConcatDataset(autofocus_datasets)
    dataloader = DataLoader(
        data, batch_size=gds[0].config["General"]["batch_size"], shuffle=True
    )
    return dataloader, b_dones


class GenDataset:
    def __init__(self, config, dataset_name, debug_max_samples=0):
        self.config = config
        self.dataset_name = dataset_name
        self.debug_max_samples = debug_max_samples
        path_images0 = os.path.join(
            config["Dataset"]["paths"]["path_dataset"],
            dataset_name,
            config["Dataset"]["paths"]["path_images"],
        )
        path_depths0 = os.path.join(
            config["Dataset"]["paths"]["path_dataset"],
            dataset_name,
            config["Dataset"]["paths"]["path_depths"],
        )
        path_segmentations0 = os.path.join(
            config["Dataset"]["paths"]["path_dataset"],
            dataset_name,
            config["Dataset"]["paths"]["path_segmentations"],
        )

        self.paths_images, self.b_done_images = get_total_paths(
            path_images0,
            config["Dataset"]["extensions"]["ext_images"],
            debug_max_samples,
        )
        self.paths_depths, self.b_done_depths = get_total_paths(
            path_depths0,
            config["Dataset"]["extensions"]["ext_depths"],
            debug_max_samples,
        )
        self.paths_segmentations, self.b_done_segmentation = get_total_paths(
            path_segmentations0,
            config["Dataset"]["extensions"]["ext_segmentations"],
            debug_max_samples,
        )
        self.b_dones = self.b_done_images, self.b_done_depths, self.b_done_segmentation
        self.b_done = all(self.b_dones)
        assert len(self.paths_images) == len(
            self.paths_depths
        ), "Different number of instances between the input and the depth maps"
        # check for segmentation
        assert len(self.paths_images) == len(
            self.paths_segmentations
        ), "Different number of instances between the input and the segmentation maps"
        assert (
            config["Dataset"]["splits"]["split_train"]
            + config["Dataset"]["splits"]["split_test"]
            + config["Dataset"]["splits"]["split_val"]
            == 1
        ), "Invalid splits (sum must be equal to 1)"

    def get_images_depths_segmentations(self):
        return (
            self.paths_images,
            self.paths_depths,
            self.paths_segmentations,
            self.b_done,
        )

    def get_splitted_dataset(self, split, num_samples=0):
        return (
            *get_splitted_dataset(
                self.config,
                split,
                self.dataset_name,
                self.paths_images,
                self.paths_depths,
                self.paths_segmentations,
                num_samples
            ),
            self.b_done,
        )


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to("cpu").float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class AutoFocusDataset(Dataset):
    """
    Dataset class for the AutoFocus Task. Requires for each image, its depth ground-truth and
    segmentation mask
    针对自动聚焦任务的数据集类。要求每幅图像及其深度真值和分割掩膜
    Args:
        :- config -: json config file
        :- dataset_name -: str
        :- split -: split ['train', 'val', 'test']
    """

    # def __init__(self, config, dataset_name, split=None, debug_max_samples=None, gd=None):
    def __init__(self, gd, split, num_samples):
        self.split = split
        self.config = gd.config
        self.dataset_name = gd.dataset_name
        assert self.split in ["train", "test", "val"], "Invalid split!"
        # if gd is None:
        #     gd = GenDataset(config, dataset_name, debug_max_samples)

        # utility func for splitting
        (
            self.paths_images,
            self.paths_depths,
            self.paths_segmentations,
            self.b_done,
        ) = gd.get_splitted_dataset(
            self.split, num_samples
        )

        # Get the transforms
        self.transform_image, self.transform_depth, self.transform_seg = get_transforms(
            self.config
        )

        # get p_flip from config
        self.p_flip = (
            self.config["Dataset"]["transforms"]["p_flip"] if split == "train" else 0
        )
        self.p_crop = (
            self.config["Dataset"]["transforms"]["p_crop"] if split == "train" else 0
        )
        self.p_rot = self.config["Dataset"]["transforms"]["p_rot"] if split == "train" else 0
        self.resize = self.config["Dataset"]["transforms"]["resize"]



    def __len__(self):
        """
        Function to get the number of images using the given list of images
        函数利用给定的图像列表得到图像的数量
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
        Getter function in order to get the triplet of images / depth maps and segmentation masks
        Getter函数以获得图像/深度图的三元组和分割掩码
        """
        #轴的个数（阶）。例如，3D 张量有 3 个轴，矩阵有 2 个轴。这在 Numpy 等 Python 库中也叫张量的 ndim
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform_image(Image.open(self.paths_images[idx]).convert("RGB"))
        depth = self.transform_depth(Image.open(self.paths_depths[idx]))
        segmentation = self.transform_seg(Image.open(self.paths_segmentations[idx]))
        # imgorig = image.clone()

        if random.random() < self.p_flip:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            segmentation = TF.hflip(segmentation)

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.resize - 1)
            max_size = self.resize - random_size
            left = int(random.random() * max_size)
            top = int(random.random() * max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            depth = TF.crop(depth, top, left, random_size, random_size)
            segmentation = TF.crop(segmentation, top, left, random_size, random_size)
            image = transforms.Resize((self.resize, self.resize))(image)
            depth = transforms.Resize((self.resize, self.resize))(depth)
            segmentation = transforms.Resize(
                (self.resize, self.resize),
                interpolation=transforms.InterpolationMode.NEAREST,
            )(segmentation)

        if random.random() < self.p_rot:
            # rotate
            random_angle = random.random() * 20 - 10  # [-10 ; 10]
            mask = torch.ones(
                (1, self.resize, self.resize)
            )  # useful for the resize at the end
            mask = TF.rotate(
                mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR
            )
            image = TF.rotate(
                image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR
            )
            depth = TF.rotate(
                depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR
            )
            segmentation = TF.rotate(
                segmentation,
                random_angle,
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            # crop to remove black borders due to the rotation
            left = torch.argmax(mask[:, 0, :]).item()
            top = torch.argmax(mask[:, :, 0]).item()
            coin = min(left, top)
            size = self.resize - 2 * coin
            image = TF.crop(image, coin, coin, size, size)
            depth = TF.crop(depth, coin, coin, size, size)
            segmentation = TF.crop(segmentation, coin, coin, size, size)
            # Resize
            image = transforms.Resize((self.resize, self.resize))(image)
            depth = transforms.Resize((self.resize, self.resize))(depth)
            segmentation = transforms.Resize(
                (self.resize, self.resize),
                interpolation=transforms.InterpolationMode.NEAREST,
            )(segmentation)
        # show([imgorig, image, depth, segmentation])
        # exit(0)
        return image, depth, segmentation
