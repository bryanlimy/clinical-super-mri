import os
import torch
import argparse
import scipy.io
import typing as t
import numpy as np
import torchio as tio
from glob import glob
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader

from supermri.data import data
from supermri.utils import utils
from supermri.models.registry import get_model


def load_args(args):
    """Loads settings from model_dir/args.json"""
    utils.load_args(args, filename=os.path.join(args.model_dir, "args.json"))
    args.input_shape = tuple(args.input_shape)
    args.scan_shape = tuple(args.scan_shape)

    # create directory for super-resolution outputs
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "sr")
    if os.path.exists(args.output_dir) and not args.overwrite:
        raise FileExistsError(f"--output_dir {args.output_dir} already exits.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def save_mat(args, filename: t.Union[Path, str], scan: torch.Tensor):
    """
    Convert a patient scan into a .mat file.

    Args:
      filename: .mat filename to be stored
      scan: scan tensor with format CHWD
    """
    assert tuple(scan.shape[1:]) == args.scan_shape
    scan = utils.to_numpy(scan)
    # rotate scan back to its original rotation
    scan = np.rot90(scan, k=-1, axes=[1, 3])
    scipy.io.savemat(
        file_name=filename,
        mdict={"FLAIRarray": scan[0], "T1array": scan[1], "T2array": scan[2]},
        do_compression=True,
    )


def main(args):
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"--input_dir {args.input_dir} not found.")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"--model_dir {args.model_dir} not found.")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    load_args(args)

    filenames = glob(os.path.join(args.input_dir, "*.mat"))
    print(f"Found {len(filenames)} .mat files in {args.input_dir}")

    model = get_model(args)
    utils.load_checkpoint(args, model=model)

    model.eval()

    for filename in filenames:
        subject = data.load_subject(
            lr_filename=filename, sequence=None, require_hr=False
        )
        sampler = tio.GridSampler(subject=subject, patch_size=args.patch_shape)
        data_loader = DataLoader(sampler, batch_size=args.batch_size)
        aggregator = tio.GridAggregator(sampler, overlap_mode="average")

        for batch in tqdm(data_loader, desc=subject.name):
            inputs, _ = data.prepare_batch(
                batch, dim=args.slice_dim, device=args.device
            )

            with torch.no_grad():
                if args.combine_sequence:
                    outputs = model(inputs)
                    if args.output_logits:
                        outputs = F.sigmoid(outputs)
                else:
                    # inference each channel separately and combine them
                    outputs = torch.zeros_like(inputs)
                    for channel in range(inputs.shape[1]):
                        channel_input = torch.unsqueeze(inputs[:, channel], dim=1)
                        channel_output = model(channel_input)
                        if args.output_logits:
                            channel_output = F.sigmoid(channel_output)
                        outputs[:, channel] = channel_output[:, 0]

            outputs = torch.unsqueeze(outputs, dim=args.slice_dim)
            aggregator.add_batch(outputs, batch[tio.LOCATION])

        output_tensor = aggregator.get_output_tensor()

        save_mat(
            args,
            filename=os.path.join(args.output_dir, f"{subject.name}.mat"),
            scan=output_tensor,
        )

    print(f"\nsaved {len(filenames)} upsampled scans to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict scans")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory with scans to upsample.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to directory with model checkpoint saved.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to directory to store the super-resolution "
        "scans. Store SR scans in input_dir/sr by default.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of samples the network process at once. "
        "By default, use the setting from the model checkpoint.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA compute.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing data in --model_dir",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        choices=[0, 1, 2],
        type=int,
        help="verbosity. 0 - no print statement, 2 - print all print statements.",
    )
    main(parser.parse_args())
