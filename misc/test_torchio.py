import os
import torch
import scipy.io
import argparse
import typing as t
import numpy as np
import torchio as tio
from tqdm import tqdm
from glob import glob
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def load_mat(filename, sequence: str = None):
    assert sequence is None or sequence in ["FLAIRarray", "T1array", "T2array"]
    data = scipy.io.loadmat(str(filename))

    if sequence is None:
        # load all sequences
        flair = data["FLAIRarray"].astype(np.float32)
        t1 = data["T1array"].astype(np.float32)
        t2 = data["T2array"].astype(np.float32)
        if np.isnan(flair).any():
            flair = np.nan_to_num(flair)
        if np.isnan(t1).any():
            t1 = np.nan_to_num(t1)
        if np.isnan(t2).any():
            t2 = np.nan_to_num(t2)
        scan = np.stack([flair, t1, t2])
    else:
        # load a particular sequence
        scan = data[sequence].astype(np.float32)
        if np.isnan(scan).any():
            scan = np.nan_to_num(scan)
        scan = np.expand_dims(scan, axis=0)

    return scan, np.eye(4)


def plot(lr_patches, hr_patches):
    lr_patches, hr_patches = lr_patches.numpy(), hr_patches.numpy()
    lr_patches = np.squeeze(lr_patches, axis=3)
    hr_patches = np.squeeze(hr_patches, axis=3)
    lr_patch, hr_patch = lr_patches[0], hr_patches[0]
    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 6), dpi=120)
    axes[0, 0].imshow(lr_patch[0], cmap="gray", interpolation="none")
    axes[0, 1].imshow(lr_patch[1], cmap="gray", interpolation="none")
    axes[0, 2].imshow(lr_patch[2], cmap="gray", interpolation="none")
    axes[1, 0].imshow(hr_patch[0], cmap="gray", interpolation="none")
    axes[1, 1].imshow(hr_patch[1], cmap="gray", interpolation="none")
    axes[1, 2].imshow(hr_patch[2], cmap="gray", interpolation="none")
    axes[1, 0].set_xlabel("FLAIR")
    axes[1, 1].set_xlabel("T1")
    axes[1, 2].set_xlabel("T2")
    figure.tight_layout()
    plt.show()


def load_subject(lr_filename: Path, sequence: str = None):
    lr = tio.ScalarImage(path=lr_filename, reader=partial(load_mat, sequence=sequence))
    hr = None
    hr_filename = Path(str(lr_filename).replace("V0", "V1"))
    if hr_filename.exists():
        hr = tio.ScalarImage(
            path=lr_filename.replace("V0", "V1"),
            reader=partial(load_mat, sequence=sequence),
        )
    name = os.path.basename(lr_filename)
    name = name[: name.find("_V0")]
    if sequence is not None:
        name += f"_{sequence}"
    return tio.Subject(lr=lr, hr=hr, name=name)


def load_subjects(filenames, combine_sequence: bool = True):
    subjects = []
    for filename in tqdm(filenames, desc="Load data"):
        if combine_sequence:
            subjects.append(load_subject(lr_filename=filename, sequence=None))
        else:
            subjects.extend(
                [
                    load_subject(lr_filename=filename, sequence=sequence)
                    for sequence in ["FLAIRarray", "T1array", "T2array"]
                ]
            )
    transform = tio.Compose(transforms=[tio.RescaleIntensity(out_min_max=(0, 1))])
    return tio.SubjectsDataset(subjects, transform=transform)


def train_val_set(args):
    filenames = glob(os.path.join(args.input_dir, "*_V0_*.mat"))
    dataset = load_subjects(filenames, combine_sequence=True)
    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=512,
        samples_per_volume=128,
        sampler=tio.UniformSampler(patch_size=(256, 1, 256)),
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    data_loader = DataLoader(queue, batch_size=8)
    for batch in data_loader:
        lr_patches = batch["lr"][tio.DATA]
        hr_patches = batch["hr"][tio.DATA]
        plot(lr_patches, hr_patches)


def test_set(args):
    filenames = glob(os.path.join(args.input_dir, "*_V0_*.mat"))[:5]
    subject = load_subject(lr_filename=filenames[0], sequence=None)

    sampler = tio.inference.GridSampler(subject=subject, patch_size=(256, 1, 256))
    data_loader = DataLoader(sampler, batch_size=8)
    aggregator = tio.GridAggregator(sampler, overlap_mode="average")

    model = torch.nn.Identity().eval()

    for batch in tqdm(data_loader):
        lr_patches = batch["lr"][tio.DATA]
        locations = batch[tio.LOCATION]
        outputs = model(lr_patches)
        aggregator.add_batch(outputs, locations)

    output_tensor = aggregator.get_output_tensor()
    target = subject["lr"][tio.DATA]
    equal = torch.equal(output_tensor, target)
    print(equal)


def main(args):
    train_val_set(args)
    test_set(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--dpi", default=120, type=int)
    main(parser.parse_args())
