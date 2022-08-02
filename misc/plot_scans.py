import os
import scipy.io
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt


def load_mat(filename):
    data = scipy.io.loadmat(str(filename))
    flair = data["FLAIRarray"].astype(np.float32)
    t1 = data["T1array"].astype(np.float32)
    t2 = data["T2array"].astype(np.float32)
    if np.isnan(flair).any():
        print(f"\t{os.path.basename(filename)}/FLAIR has NaN values")
        flair = np.nan_to_num(flair)
    if np.isnan(t1).any():
        print(f"\t{os.path.basename(filename)}/T1 has NaN values")
        t1 = np.nan_to_num(t1)
    if np.isnan(t2).any():
        print(f"\t{os.path.basename(filename)}/T2 has NaN values")
        t2 = np.nan_to_num(t2)
    scan = np.stack([flair, t1, t2])
    scan = np.transpose(scan, axes=[3, 0, 1, 2])
    return scan


def main(args):
    filenames = glob(os.path.join(args.input_dir, "*.mat"))
    for filename in tqdm(filenames):
        scan = load_mat(filename)
        slices = range(len(scan)) if args.all_slices else [59, 65, 68, 74, 76]
        for slice in slices:
            figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), dpi=args.dpi)
            axes[0].imshow(scan[slice, 0], cmap="gray", interpolation="none")
            axes[0].set_xlabel("FLAIR")
            axes[1].imshow(scan[slice, 1], cmap="gray", interpolation="none")
            axes[1].set_xlabel("T1")
            axes[1].set_title(f"Slice {slice}")
            axes[2].imshow(scan[slice, 2], cmap="gray", interpolation="none")
            axes[2].set_xlabel("T2")
            plt.setp(axes, xticks=[], yticks=[])
            figure.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.tight_layout()
            plt.savefig(filename.replace(".mat", f"_slice{slice}.pdf"), dpi=args.dpi)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--all_slices", action="store_true")
    parser.add_argument("--dpi", default=120, type=int)
    main(parser.parse_args())
