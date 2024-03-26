import os
import re
import random
import albumentations as A
import numpy as np
import cv2
import imageio
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms as T
import h5py

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset, DataLoader

import utils as u
from einops import rearrange, repeat


class DispDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            data_dir,
            filename_path,
            crop_size,
            read_fn,
            fixed_focal = 725.0087,
            scale = 100,
            canonical_focal = 700
            ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.filename_path = os.path.join('/home/xiangmochu/Accelerator-Simple-Template-depth_est_diff/datafiles', filename_path)
        self.filename_list = self.read_txtfiles(self.filename_path)
        self.crop_size = crop_size
        self.fixed_focal = fixed_focal
        self.scale = scale
        self.canonical_focal = canonical_focal

        self.basic_transformation = [
            A.HorizontalFlip(),
            # A.AdvancedBlur(blur_limit=(3, 31)),
            # A.MotionBlur(blur_limit=(3, 31)),
            A.RandomCrop(self.crop_size[0], self.crop_size[1]),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]

        self.geo_transformation = [
            A.Rotate(limit=30, p=0.1),
        ]

        additional_targets = {'depth': 'mask'}
        self.aug = A.Compose(transforms=self.basic_transformation,
                        additional_targets=additional_targets)
        
        self.geo_aug = A.Compose(transforms=self.geo_transformation,
                        additional_targets=additional_targets)
        
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.read_data = load_read_fn(read_fn)

    def read_txtfiles(self, txt_path):
        with open(txt_path) as f:
            lines = f.readlines()

        lines = [l.strip().split(',') for l in lines]
        return lines

    def __len__(self):
        return len(self.filename_list)

    def augument(self, image, depth):
        rdm = random.random() * 0.5 + 1 # [1, 1.5]
        geo_augmented = self.geo_aug(image=image, depth=depth)
        image = geo_augmented['image']
        depth = geo_augmented['depth']

        image, focal_ratio = fit_resize(image, self.crop_size, cv2.INTER_LINEAR, rdm)
        if image.shape[-1] == 4:
            image = image[..., :3]
        depth, _ = fit_resize(depth, self.crop_size, cv2.INTER_NEAREST,rdm)

        augmented = self.aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        image = self.to_tensor(image)
        image = self.normalize(image)
        # if self.depth_format == 'metric':
        #     depth = self.to_tensor(depth) / focal_ratio
        # else:
        #     depth = self.to_tensor(depth)
        return image, depth
    
    def check_depth(self, depth):
        if (depth < 1e-3).astype(float).mean() > 0.995:
            raise Exception('Depth map contains too many invalid values')
        # if depth[depth>0].std() < 0.05:
        #     raise Exception('Depth map has low variance')
    
    def __getitem__(self, idx):
        metadata = {'dataset_name': self.dataset_name,
                    'image_path': self.filename_list[idx][0],
                    'depth_path': self.filename_list[idx][1]}
        try:
            image, depth = self.read_data(self, idx) 
            image, depth = self.augument(image, depth)
            depth[depth > 100] = 100
            self.check_depth(depth)
            # depth[depth > depth.mean() + 3*depth.std()] = 0

            depth = depth / 100 # [0, 1]
            # depth = repeat(depth, 'h w -> 3 h w')

            return {'image': image, 'depth': depth, 
                    'metadata': metadata}
        except Exception as e:
            # raise error and give metadata
            raise Exception(f'Error in reading {self.dataset_name} dataset: {e} \n {metadata}')


class DataFeeder:
    def __init__(self, cfg):
        self.cfg = cfg
        data_group_name = cfg.data_group_by_gpu[cfg.local_rank]
        self.data_group_name = data_group_name
        selected_group = cfg.data.data_groups[data_group_name].datasets
        crop_size = cfg.data.data_groups[data_group_name].crop_size
        crop_size = (crop_size.h, crop_size.w)

        for dataset_name, dataset_cfg in cfg.data.datasets.items():
            dataset_cfg['crop_size'] = crop_size
        datasets = {dataset_name: DispDataset(**cfg.data.datasets[dataset_name]) for dataset_name in selected_group}
        self.dataset = ConcatDataset(datasets.values())

        group_counter = Counter(cfg.data_group_by_gpu)
        num_replicas = group_counter[data_group_name]
        data_rank = u.get_rank_by_category(cfg.data_group_by_gpu, cfg.local_rank)
        print('num_replicas:', num_replicas, 'data_rank:', data_rank, 'local_rank:', cfg.local_rank)
        self.data_sampler = DistributedSampler(dataset=self.dataset, num_replicas=num_replicas, rank=data_rank)
        self.data_sampler.set_epoch(0)
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=cfg.batch_size,
            shuffle=False, sampler=self.data_sampler, num_workers=cfg.num_workers,
            pin_memory=True, drop_last=True)

        self.data_length = len(self.data_loader)
        self.data_iter = iter(self.data_loader)
        self.iter_counter = 0
        self.epoch_counter = 0

        self.log_path = os.path.join(cfg.log_path, 'data_error.log')
    
    def __call__(self):
        while True:
            try:
                inputs = next(self.data_iter)
                self.set_counter()
                return inputs
            except Exception as e:
                # print('Failed to get data', e)
                with open(self.log_path, 'a') as f:
                    f.write('LocakRank:' + str(self.cfg.local_rank) + '\n')
                    f.write('DataGroup:' + str(self.data_group_name) + '\n')
                    f.write(str(e) + '\n')
                continue
    
    def init_epoch(self, epoch):
        self.epoch_counter = epoch
        self.data_sampler.set_epoch(epoch)
        self.data_iter = iter(self.data_loader)
        self.iter_counter = 0
    
    def set_counter(self):
        self.iter_counter += 1
        if self.iter_counter >= self.data_length:
            self.iter_counter = 0
            self.epoch_counter += 1
            self.data_sampler.set_epoch(self.epoch_counter)
            del self.data_loader
            self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.cfg.batch_size,
                shuffle=False, sampler=self.data_sampler, num_workers=self.cfg.num_workers,
                pin_memory=True, drop_last=True)
            self.data_iter = iter(self.data_loader)


def fit_resize(img, crop_size, interpolation, rdm=1):
    '''
    resize image to crop the largest region
    '''
    h, w = img.shape[:2]
    ratio_h = crop_size[0] / h * rdm
    ratio_w = crop_size[1] / w * rdm
    if ratio_h > ratio_w:
        new_h = crop_size[0]
        new_w = int(w * ratio_h)
        focal_ratio = ratio_h
    else:
        new_h = int(h * ratio_w)
        new_w = crop_size[1]
        focal_ratio = ratio_w
    new_h = max(new_h, crop_size[0])
    new_w = max(new_w, crop_size[1])
    return cv2.resize(img, (new_w, new_h), interpolation), focal_ratio


def read_affine(self, idx):
    image_path, depth_path, *_ = self.filename_list[idx]
    image = imageio.imread(os.path.join(self.data_dir, image_path))
    depth = imageio.imread(os.path.join(self.data_dir, depth_path))
    depth = affine_norm(depth)
    return image, depth


def read_affine_inv(self, idx):
    image_path, depth_path, *_ = self.filename_list[idx]
    image = imageio.imread(os.path.join(self.data_dir, image_path))
    depth = imageio.imread(os.path.join(self.data_dir, depth_path))
    depth = depth.max() - depth
    depth = affine_norm(depth)
    return image, depth


def read_scale(self, idx):
    image_path, depth_path, *_ = self.filename_list[idx]
    image = imageio.imread(os.path.join(self.data_dir, image_path))
    depth = imageio.imread(os.path.join(self.data_dir, depth_path))
    depth[depth == depth.max()] = 0
    h,w = image.shape[:2]
    h_, w_ = depth.shape[:2]
    if h != h_ or w != w_:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = scale_norm(depth)
    return image, depth


def read_metric(self, idx):
    '''read metric depth with known focal length'''
    image_path, depth_path, focal = self.filename_list[idx]
    focal = float(focal)
    image = imageio.imread(os.path.join(self.data_dir, image_path))
    depth = imageio.imread(os.path.join(self.data_dir, depth_path)).astype(float)
    # depth[depth == depth.max()] = 0
    h,w = image.shape[:2]
    h_, w_ = depth.shape[:2]
    if h != h_ or w != w_:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = depth / self.scale / focal * self.canonical_focal
    return image, depth


def read_metric_fixed_focal(self, idx):
    '''read metric depth with fixed focal length'''
    image_path, depth_path = self.filename_list[idx]
    focal = self.fixed_focal
    image = imageio.imread(os.path.join(self.data_dir, image_path))
    depth = imageio.imread(os.path.join(self.data_dir, depth_path))
    # depth[depth == depth.max()] = 0
    h,w = image.shape[:2]
    h_, w_ = depth.shape[:2]
    if h != h_ or w != w_:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = depth / self.scale / focal * self.canonical_focal
    return image, depth


def read_presil(self, idx):
    pass


def read_vkitti(self, idx):
    pass


def read_pfm(path):
    """Read pfm file.
    Args:
        path (str): path to file
    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def read_bin(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def affine_norm(depth: np.ndarray):
    non_zero = depth > 0
    bottom = np.percentile(depth[non_zero], 5)
    top = np.percentile(depth[non_zero], 95)
    
    # [5%, 95%] -> [0.05, 0.95]
    depth = (depth-bottom)/(top-bottom)*0.9 + 0.05
    depth = np.clip(depth, 1e-4, 1)
    depth *= non_zero
    return depth


def scale_norm(depth: np.ndarray):
    non_zero = depth > 0
    mu = np.median(depth[non_zero])
    depth = depth / (mu + 1e-6) 
    return depth


def load_read_fn(fn_name: str):
    assert fn_name in globals(), 'No such function: ' + fn_name
    return eval(fn_name)