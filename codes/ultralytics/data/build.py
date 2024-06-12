# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.loaders import (
    LOADERS,
    LoadImages,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
    Load_Seg_Img,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from .dataset import YOLODataset
from .utils import PIN_MEMORY
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
import time

class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, n_tasks=1, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        self.n_tasks = n_tasks
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
        

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()

    def shuffle_sampler(self):
        # data_source = self.sampler.dataset
        data_source = self.sampler.data_source
        im_files = data_source.im_files
        label_files = data_source.label_files
        labels = data_source.labels
        random.seed(int(time.time()))

        camera_groups = defaultdict(list)
        index_map = list(range(len(im_files)))
        for i, file in enumerate(im_files):
            filename = os.path.basename(file)
            camera_number = int(filename.split("_camera")[1][0])
            if camera_number <= self.n_tasks:
                camera_groups[camera_number].append(i)  # Store index instead of file name
        # Randomly sort files for each camera group
        shuffled_indices = []
        camera_keys = list(camera_groups.keys())
        random.shuffle(camera_keys)
        for key in camera_keys:
            group_indices = camera_groups[key]
            random.shuffle(group_indices)
            shuffled_indices.extend(group_indices)
        # Update the order of files in the data source
        new_order = [index_map[i] for i in shuffled_indices]
        data_source.im_files = [im_files[i] for i in new_order]
        data_source.labels = [labels[i] for i in new_order]
        data_source.label_files = [label_files[i] for i in new_order]




    # def _get_iterator(self):
    #     self.shuffle_sampler()  # Make sure it's randomized every time you get the iterator
    #     return super().__iter__()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1,n_tasks=1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        n_tasks=n_tasks,
    )


def check_source(source):
    """Check source type and return corresponding flag values."""
    def is_segments_info(src):
        return len(src) == 5 and isinstance(src[0], list) and all(isinstance(item, str) for item in src[0])
        
    webcam, screenshot, from_img, in_memory, tensor, seg_info  = False, False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        if is_segments_info(source):  # Used to check whether source conforms to the structure of segments_info
            seg_info = True
            # tensor = True
        else:
            source = autocast_list(source)  # convert all list elements to PIL or np arrays
            from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor, seg_info


def load_inference_source(source=None, imgsz=640, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        imgsz (int, optional): The size of the image for inference. Default is 640.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, webcam, screenshot, from_img, in_memory, tensor, seg_info = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif webcam:
        dataset = LoadStreams(source, imgsz=imgsz, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source, imgsz=imgsz)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz)
    elif seg_info:
        dataset = Load_Seg_Img(source, imgsz=imgsz, vid_stride=vid_stride)
    else:
        dataset = LoadImages(source, imgsz=imgsz, vid_stride=vid_stride)

    # Attach source types to the dataset
    # tensor = True
    # source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)
    setattr(dataset, "source_type", source_type)
    print("")
    return dataset
