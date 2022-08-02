import os
import scipy.io
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob


def load_mat(filename: str) -> np.ndarray:
    """Load filename and return scan array in format CHWD

    Channel dimension has order [FLAIR, T1, T2]
    """
    data = scipy.io.loadmat(filename)
    flair = data["FLAIRarray"].astype(np.float32)
    t1 = data["T1array"].astype(np.float32)
    t2 = data["T2array"].astype(np.float32)
    return np.stack([flair, t1, t2])


def main(args):
    filenames = glob(os.path.join(args.input_dir, "*.mat"))

    print(f"found {len(filenames)} .mat files in {args.input_dir}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in tqdm(filenames):
        scan = load_mat(filename)

        # replace NaN values with zeros
        if np.isnan(scan).any():
            scan = np.nan_to_num(scan)

        # replace negative values with zeros
        # p.s. scans are supposed to already be in (0, 1)
        scan = np.maximum(scan, 0.0)

        basename = os.path.basename(filename).replace(".mat", ".npy")
        np.save(os.path.join(args.output_dir, basename), scan)

    print(f"{len(filenames)} .npy files saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    main(parser.parse_args())
