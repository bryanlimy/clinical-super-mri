import os
import torch
import random
import scipy.io
import typing as t
import numpy as np
import torchio as tio
from glob import glob
from functools import partial
from torch.utils.data import DataLoader


def get_scan_shape(filename: str):
    extension = filename[:-3]
    if extension == "mat":
        data = scipy.io.loadmat(filename)
        shape = data["FLAIRarray"].shape
    else:
        scan = np.load(filename)
        shape = scan.shape[1:]
    shape = list(shape)
    # rotation swap first and third index
    shape[0], shape[2] = shape[2], shape[0]
    return shape


def rotate(scan: np.ndarray):
    """rotate scan 90 degree to the left"""
    assert len(scan.shape) == 4
    # copy is needed to avoid negative strides error
    return np.rot90(scan, k=1, axes=[1, 3]).copy()


def load_mat(filename: str, sequence: str = None):
    """
    Reader callable object for tio.ScalarImage to load sequence from mat file.

    Note: NaN and negative values are replaced with zeros.

    Args:
      filename: path to the mat file
      sequence: FLAIRarray, T1array or T2array to load specify sequence, or
                None to load all sequences as channels.

    Returns:
      scan: np.ndarray in shape CHWD
      affine: 4x4 affine matrix
    """
    sequences = ["FLAIRarray", "T1array", "T2array"]
    assert sequence is None or sequence in sequences

    data = scipy.io.loadmat(filename)

    if sequence is None:
        # load all sequences
        flair = data["FLAIRarray"].astype(np.float32)
        t1 = data["T1array"].astype(np.float32)
        t2 = data["T2array"].astype(np.float32)
        scan = np.stack([flair, t1, t2])
    else:
        # load a particular sequence
        scan = data[sequence].astype(np.float32)
        scan = np.expand_dims(scan, axis=0)
    # replace NaN values with zeros
    if np.isnan(scan).any():
        scan = np.nan_to_num(scan)
    # replace negative values with zeros
    scan = np.maximum(scan, 0.0)
    # rotate scan to the left 90 degree
    scan = rotate(scan)
    return scan, np.eye(4)


def load_npy(filename: str, sequence: str = None):
    """
    Reader callable object for tio.ScalarImage to load sequence from npy file.

    Args:
      filename: path to the npy file
      sequence: FLAIRarray, T1array or T2array to load specify sequence, or
                None to load all sequences as channels.

    Returns:
      scan: np.ndarray in shape CHWD
      affine: 4x4 affine matrix
    """
    sequences = ["FLAIRarray", "T1array", "T2array"]
    assert sequence is None or sequence in sequences
    scan = np.load(filename)
    if sequence is not None:
        # load a particular sequence
        channel = sequences.index(sequence)
        scan = np.expand_dims(scan[channel], axis=0)
    # rotate scan to the left 90 degree
    scan = rotate(scan)
    return scan, np.eye(4)


def load_subject(lr_filename: str, sequence: str = None, require_hr: bool = False):
    extension = lr_filename[-3:]
    assert extension in [
        "npy",
        "mat",
    ], "scans must be in npy or mat format, got {extension}."
    reader = partial(load_mat if extension == "mat" else load_npy, sequence=sequence)

    lr = tio.ScalarImage(path=lr_filename, reader=reader)

    hr, hr_filename = None, lr_filename.replace("V0", "V1")
    hr_exists = os.path.exists(hr_filename)
    if require_hr and not hr_exists:
        raise FileNotFoundError(f"{hr_filename} not found.")
    if hr_exists:
        hr = tio.ScalarImage(path=hr_filename, reader=reader)

    name = os.path.basename(lr_filename)
    if ".npy" in name or ".mat" in name:
        name = name[:-4]
    if "_V0" in name:
        name = name[: name.find("_V0")]
    if sequence is not None:
        name += f"_{sequence}"

    if hr_exists:
        subject = tio.Subject(lr=lr, hr=hr, name=name)
    else:
        subject = tio.Subject(lr=lr, name=name)

    return subject


def load_dataset(filenames: t.List[str], combine_sequence: bool = True):
    subjects = []
    for filename in filenames:
        if combine_sequence:
            subjects.append(
                load_subject(lr_filename=filename, sequence=None, require_hr=True)
            )
        else:
            subjects.extend(
                [
                    load_subject(
                        lr_filename=filename, sequence=sequence, require_hr=True
                    )
                    for sequence in ["FLAIRarray", "T1array", "T2array"]
                ]
            )
    return tio.SubjectsDataset(subjects)


def get_loaders(
    args, val_ratio: float = 0.2, test_ratio: float = 0.1, random_state: int = 42
):
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"{args.input_dir} not found.")

    filenames = sorted(glob(os.path.join(args.input_dir, f"*_V0*.{args.extension}")))

    assert len(filenames) > 0, f"no {args.extension} files found in {args.input_dir}"
    print(f"found {len(filenames)} {args.extension} pairs in {args.input_dir}")

    # shuffle files
    random.Random(random_state).shuffle(filenames)

    args.ds_name = None
    if "warp" in args.input_dir:
        args.ds_name = "warp"
    elif "affine" in args.input_dir:
        args.ds_name = "affine"
    elif "rigid" in args.input_dir:
        args.ds_name = "rigid"
    args.scan_shape = get_scan_shape(filenames[0])
    args.scan_types = ["FLAIR", "T1", "T2"]

    test_size = int(len(filenames) * test_ratio)
    val_size = int(len(filenames) * val_ratio)

    args.test_filenames = filenames[:test_size]
    val_filenames = filenames[test_size : test_size + val_size]
    train_filenames = filenames[test_size + val_size :]

    C = 3 if args.combine_sequence else 1
    # calculate HWD dimension, dim indicate the dimension to insert 1
    dim = 0 if args.ds_name == "warp" else 1
    if args.patch_size is None:
        if args.ds_name == "warp":
            H, W = args.scan_shape[1], args.scan_shape[2]
            args.n_patches = args.scan_shape[0]
        else:
            H, W = args.scan_shape[0], args.scan_shape[2]
            args.n_patches = args.scan_shape[1]
    else:
        assert args.n_patches is not None, "--n_patches is not defined."
        H, W = args.patch_size, args.patch_size

    # TorchIO sampler sample 3D patches from image
    # we insert 1 to patch_size to output 2D patches effectively
    patch_size = [H, W]
    patch_size.insert(dim, 1)

    args.patch_shape = tuple(patch_size)
    args.slice_dim = dim + 2  # slice dimension in NCHWD
    args.input_shape = (C, H, W)

    train_dataset = load_dataset(
        train_filenames, combine_sequence=args.combine_sequence
    )
    train_queue = tio.Queue(
        subjects_dataset=train_dataset,
        max_length=args.n_patches * 4,
        samples_per_volume=args.n_patches,
        sampler=tio.UniformSampler(patch_size=args.patch_shape),
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    train_loader = DataLoader(train_queue, batch_size=args.batch_size, pin_memory=True)

    val_dataset = load_dataset(val_filenames, combine_sequence=args.combine_sequence)
    val_queue = tio.Queue(
        subjects_dataset=val_dataset,
        max_length=args.n_patches * 4,
        samples_per_volume=args.n_patches,
        sampler=tio.UniformSampler(patch_size=args.patch_shape),
        num_workers=args.num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )
    val_loader = DataLoader(val_queue, batch_size=args.batch_size, pin_memory=True)

    return train_loader, val_loader


def prepare_batch(
    batch, dim: int, device: torch.device = "cpu"
) -> (torch.Tensor, torch.Tensor):
    """
    Prepare a batch from TorchIO data loader

    Args:
      batch: a batch from data loader
      dim: slice dimension in NCHWD
      device: torch device

    Returns:
      lr: low resolution batch
      hr: high resolution batch if exists, else None
    """

    def prepare(tensor):
        tensor = torch.squeeze(tensor, dim=dim)
        return tensor.to(device)

    lr, hr = batch["lr"][tio.DATA], None
    lr = prepare(lr)
    if "hr" in batch:
        hr = batch["hr"][tio.DATA]
        hr = prepare(hr)
    return lr, hr


def random_samples(args, val_loader, num_samples: int = 6):
    """Randomly select num_samples samples from val_loader for plotting"""
    batch = next(iter(val_loader))
    inputs, targets = prepare_batch(batch, dim=args.slice_dim, device=args.device)
    indexes = np.random.choice(
        inputs.shape[0], size=min(num_samples, inputs.shape[0]), replace=False
    )
    return inputs[indexes], targets[indexes]
