import io
import torch
import platform
import warnings
import matplotlib
import numpy as np
import typing as t
from PIL import Image
import seaborn as sns
from pathlib import Path
from typing import Union
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from supermri.utils import utils

warnings.simplefilter("ignore", UserWarning)

JET, BWR = cm.get_cmap("jet"), cm.get_cmap("bwr")
COLORMAP = JET(np.arange(256))[:, :3]


def normalize(inputs: np.ndarray):
    """normalize array x to [0, 1]"""
    return np.maximum(inputs, 0) / np.max(inputs)


def set_xticks(
    axis,
    ticks_loc: np.ndarray,
    ticks: t.Union[np.ndarray, list] = None,
    label: str = "",
):
    axis.set_xticks(ticks_loc)
    if ticks is None:
        ticks = ticks_loc.astype(int)
    axis.set_xticklabels(ticks)
    if label:
        axis.set_xlabel(label)


def set_yticks(
    axis,
    ticks_loc: np.ndarray,
    ticks: t.Union[np.ndarray, list] = None,
    label: str = "",
):
    axis.set_yticks(ticks_loc)
    if ticks is None:
        ticks = ticks_loc.astype(int)
    axis.set_yticklabels(ticks)
    if label:
        axis.set_ylabel(label)


def set_tick_params(axis, length: int = 1):
    axis.tick_params(axis="both", which="both", length=length)


def remove_top_right_spines(axis: matplotlib.axes.Axes):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def remove_spines(axis: matplotlib.axes.Axes):
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_ticks(axis: matplotlib.axes.Axes):
    """Remove x and y ticks in axis"""
    axis.set_xticks([])
    axis.set_yticks([])


class Summary:
    """
    TensorBoard Summary class to log model training performance and record
    generated samples

    By default, training summary are stored under args.output_dir and
    validation summary are stored at args.output_dir/validation
    """

    def __init__(self, args):
        self.dpi = args.dpi
        # save plots and figures to disk
        self.save_plots = args.save_plots
        if self.save_plots:
            self.plot_dir = args.output_dir / "plots"
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            self.format = "pdf"

        self.scan_types = args.scan_types

        self.writers = [
            SummaryWriter(args.output_dir),
            SummaryWriter(args.output_dir / "validation"),
            SummaryWriter(args.output_dir / "test"),
        ]

        # matplotlib settings
        sns.set_style("white")
        plt.style.use("seaborn-deep")
        if args.verbose == 2 and platform.system() == "Darwin":
            matplotlib.use("TkAgg")

    def get_writer(self, mode: int):
        """get writer for the specified mode
        Args:
          mode: int, 0 - train, 1 - validation, 2 - test
        Returns:
          writer
        """
        assert mode in [0, 1, 2], f"No writer with mode {mode}"
        return self.writers[mode]

    def save_figure(
        self, filename: Union[str, Path], figure: t.Type[plt.Figure], close: bool = True
    ):
        """Save matplotlib figure to self.plot_dir/filename"""
        fname = self.plot_dir / f"{filename}.{self.format}"
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            str(fname),
            dpi=self.dpi,
            bbox_inches="tight",
            pad_inches=0.01,
            transparent=True,
        )
        if close:
            plt.close(figure)

    def flush(self):
        """Flush all writers"""
        for writer in self.writers:
            writer.flush()

    def close(self):
        """Close all writers"""
        self.flush()
        for writer in self.writers:
            writer.close()

    def scalar(self, tag, value, step: int = 0, mode: int = 0):
        """Write scalar value to summary
        Args:
          tag: data identifier
          value: scalar value
          step: global step value to record
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        writer = self.get_writer(mode)
        writer.add_scalar(tag, value, global_step=step)
        writer.flush()

    def histogram(self, tag, value, step=0, mode: int = 0):
        """Write histogram to summary
        Args:
          tag: data identifier
          value: values to build histogram
          step: global step value to record
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        writer = self.get_writer(mode)
        writer.add_histogram(tag, values=value, global_step=step)

    def image(self, tag, value, step=0, dataformats: str = "CHW", mode: int = 0):
        """Write image to summary

        Note: TensorBoard only accept images with value within [0, 1)

        Args:
          tag: data identifier
          value: image array dataformats
          dataformats: image data format, e.g. (CHW) or (NCHW)
          step: global step value to record
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        assert len(value.shape) in [3, 4] and len(value.shape) == len(dataformats)
        writer = self.get_writer(mode)
        if len(dataformats) == 3:
            writer.add_image(tag, value, global_step=step, dataformats=dataformats)
        else:
            writer.add_images(tag, value, global_step=step, dataformats=dataformats)

    def figure(self, tag, figure, step: int = 0, close: bool = True, mode: int = 0):
        """Write matplotlib figure to summary
        Args:
          tag: data identifier
          figure: matplotlib figure or a list of figures
          step: global step value to record
          close: flag to close figure
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        buffer = io.BytesIO()
        figure.savefig(
            buffer, dpi=self.dpi, format="png", bbox_inches="tight", pad_inches=0.02
        )
        buffer.seek(0)
        image = Image.open(buffer)
        image = transforms.ToTensor()(image)
        self.image(tag, image, step=step, dataformats="CHW", mode=mode)
        if self.save_plots:
            self.save_figure(f"epoch_{step:03d}/{tag}", figure=figure, close=False)
        if close:
            plt.close()

    def plot_side_by_side(self, tag: str, samples: dict, step: int = 0, mode: int = 0):
        """Plot side by side comparison in a grid for each channel

        Args:
          tag: name of the plot in TensorBoard
          samples: inputs, targets and outputs in shape (NCHW)
            dictionary in {'inputs': tensor, 'targets': tensor, 'outputs': tensor}
          step: the current step or epoch
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        samples = {k: utils.to_numpy(v) for k, v in samples.items()}
        shape = samples["inputs"].shape
        figsize = (8, 3) if shape[1] == 1 else (7.5, 2.5 * shape[1])
        for patch in range(shape[0]):
            figure, axes = plt.subplots(
                nrows=shape[1],
                ncols=3,
                sharex=True,
                sharey=True,
                squeeze=False,
                gridspec_kw={"wspace": 0.05, "hspace": 0.05},
                figsize=figsize,
                dpi=self.dpi,
            )
            for channel in range(shape[1]):
                # extract images for current channel:
                input_image = samples["inputs"][patch, channel]
                output_image = samples["outputs"][patch, channel]
                target_image = samples["targets"][patch, channel]

                kwargs = {"cmap": "gray", "interpolation": "none"}
                axes[channel, 0].imshow(input_image, **kwargs)
                axes[channel, 1].imshow(output_image, **kwargs)
                axes[channel, 2].imshow(target_image, **kwargs)
                # show scan type if scan is multi-channel
                if shape[1] > 1:
                    axes[channel, 0].set_ylabel(self.scan_types[channel])
                # set title for top row only
                if channel == 0:
                    axes[channel, 0].set_title("input")
                    axes[channel, 1].set_title("generated")
                    axes[channel, 2].set_title("target")

            self.figure(f"{tag}/patch_#{patch:03d}", figure, step=step, mode=mode)

    def plot_difference_maps(
        self, tag: str, samples: dict, step: int = 0, mode: int = 0
    ):
        """Plot difference maps in a grid for each channel
        Args:
          tag: name of the plot in TensorBoard
          samples: inputs, targets and outputs in shape (NCHW)
            dictionary in {'inputs': tensor, 'targets': tensor, 'outputs': tensor}
          step: the current step or epoch
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        samples = {k: utils.to_numpy(v) for k, v in samples.items()}
        shape = samples["inputs"].shape
        figsize = (7.5, 2.5) if shape[1] == 1 else (8, 2.5 * shape[1])

        for p in range(shape[0]):
            figure, axes = plt.subplots(
                nrows=shape[1],
                ncols=4,
                gridspec_kw={
                    "width_ratios": [1, 1, 1, 0.05],
                    "hspace": 0.05,
                    "wspace": 0.0,
                },
                figsize=figsize,
                squeeze=False,
                dpi=self.dpi,
            )
            for c in range(shape[1]):
                t_i = samples["targets"][p, c] - samples["inputs"][p, c]
                i_o = samples["inputs"][p, c] - samples["outputs"][p, c]
                t_o = samples["targets"][p, c] - samples["outputs"][p, c]

                vmin, vmax = np.min([t_i, i_o, t_o]), np.max([t_i, i_o, t_o])
                vabs = np.max([np.abs(vmin), vmax])
                kwargs = {
                    "cmap": "bwr",
                    "interpolation": "none",
                    "aspect": "equal",
                    "vmin": -vabs,
                    "vmax": vabs,
                }
                axes[c, 0].imshow(t_i, **kwargs)
                axes[c, 1].imshow(i_o, **kwargs)
                axes[c, 2].imshow(t_o, **kwargs)

                norm = colors.Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
                figure.colorbar(cm.ScalarMappable(norm=norm, cmap=BWR), cax=axes[c, 3])

                if shape[1] > 1:
                    axes[c, 0].set_ylabel(self.scan_types[c])
                for i in range(3):
                    plt.setp(axes[c, i], xticks=[], yticks=[])

            axes[0, 0].set_title("target - input")
            axes[0, 1].set_title("input - generated")
            axes[0, 2].set_title("target - generated")

            self.figure(f"{tag}/patch_#{p:03d}", figure, step=step, mode=mode)

    def plot_stitched(
        self,
        tag: str,
        samples: dict,
        dim: int,
        slice: int = None,
        step: int = 0,
        mode: int = 2,
    ):
        """Plot stitched side by side comparison

        Args:
          tag: name of the plot in TensorBoard
          samples: dictionary with keys inputs, targets and outputs
                    each tensor should have format CHWD
          dim: dimension to slice
          slice: slice to plot, plot middle slice if None.
          step: the current step or epoch
          mode: mode of the data. 0 - train, 1 - validation, 2 - test
        """
        if slice is None:
            slice = samples["inputs"].shape[dim] // 2

        # rearrange slice dimension to front
        axes = list(range(len(samples["inputs"].shape)))
        axes.pop(dim)
        axes.insert(0, dim)

        def convert(tensor):
            array = utils.to_numpy(tensor)
            array = np.transpose(array, axes=axes)
            return array[slice]

        samples = {k: convert(v) for k, v in samples.items()}

        shape = samples["inputs"].shape

        figure, axes = plt.subplots(
            nrows=shape[0],
            ncols=3,
            figsize=(8, 3) if shape[0] == 1 else (7.5, 2.5 * shape[0]),
            squeeze=False,
            dpi=self.dpi,
        )
        for channel in range(shape[0]):
            # extract images for current channel:
            input_image = samples["inputs"][channel]
            output_image = samples["outputs"][channel]
            target_image = samples["targets"][channel]

            kwargs = {"cmap": "gray", "interpolation": "none"}
            axes[channel, 0].imshow(input_image, **kwargs)
            axes[channel, 1].imshow(output_image, **kwargs)
            axes[channel, 2].imshow(target_image, **kwargs)
            # show scan type if scan is multi-channel
            if shape[0] > 1:
                axes[channel, 0].set_ylabel(self.scan_types[channel])
            # set title for top row only
            if channel == 0:
                axes[channel, 0].set_title("input")
                axes[channel, 1].set_title("generated")
                axes[channel, 2].set_title("target")

        plt.setp(axes, xticks=[], yticks=[])
        figure.subplots_adjust(wspace=0.05, hspace=0.05)
        self.figure(f"{tag}/slice_#{slice:03d}", figure, step=step, mode=mode)

    def plot_attention_masks(
        self,
        tag: str,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        gate_inputs: dict,
        gate_masks: dict,
        step: int = 0,
        mode: int = 1,
    ):
        """plot attention masks from AG-UNet"""
        assert inputs.shape == outputs.shape
        assert sorted(gate_inputs.keys()) == sorted(gate_masks.keys())
        inputs, outputs = utils.to_numpy(inputs), utils.to_numpy(outputs)
        if len(inputs.shape) == 3:
            # average all channels
            inputs, outputs = np.mean(inputs, axis=0), np.mean(outputs, axis=0)

        aspect, num_images = "equal", len(gate_inputs) + 2
        figure, axes = plt.subplots(
            nrows=num_images,
            ncols=1,
            gridspec_kw={"hspace": 0.125, "wspace": 0.1},
            figsize=(5, 4 * num_images),
            dpi=self.dpi,
        )

        axes[0].imshow(inputs, cmap="gray", aspect=aspect)
        set_xticks(axis=axes[0], ticks_loc=np.linspace(0, inputs.shape[1] - 1, 5))
        set_yticks(
            axis=axes[0],
            ticks_loc=np.linspace(0, inputs.shape[0] - 1, 5),
            label=f"Input",
        )
        set_tick_params(axis=axes[0])

        row = 1
        for i in range(len(gate_inputs)):
            gate_input = utils.to_numpy(torch.mean(gate_inputs[f"gate_{i}"][0], dim=0))
            gate_mask = utils.to_numpy(gate_masks[f"gate_{i}"][0, 0])
            heatmap = COLORMAP[np.uint8(255.0 * gate_mask)]
            superimpose = 0.6 * heatmap + 0.4 * gate_input[..., np.newaxis]
            axes[row].imshow(normalize(superimpose), aspect=aspect)
            set_xticks(
                axis=axes[row], ticks_loc=np.linspace(0, superimpose.shape[1] - 1, 5)
            )
            set_yticks(
                axis=axes[row],
                ticks_loc=np.linspace(0, superimpose.shape[0] - 1, 5),
                label=f"AG{i+1}",
            )
            set_tick_params(axis=axes[row])
            row += 1

        axes[row].imshow(outputs, cmap="gray", aspect=aspect)
        set_xticks(axis=axes[row], ticks_loc=np.linspace(0, outputs.shape[1] - 1, 5))
        set_yticks(
            axis=axes[row],
            ticks_loc=np.linspace(0, outputs.shape[0] - 1, 5),
            label=f"Output",
        )
        set_tick_params(axis=axes[row])

        self.figure(tag, figure, step=step, mode=mode)

    def plot_gradcam(
        self,
        tag: str,
        images: torch.Tensor,
        cams: torch.Tensor,
        titles: t.List[str] = None,
        step: int = 0,
        mode: int = 1,
    ):
        if len(images.shape) == 4:
            images = torch.mean(images, dim=1)

        assert images.shape == cams.shape
        images, cams = utils.to_numpy(images), utils.to_numpy(cams)

        for i in range(len(images)):
            figure, axes = plt.subplots(
                nrows=1,
                ncols=2,
                gridspec_kw={"width_ratios": [1, 0.05], "hspace": 0.1, "wspace": 0.05},
                figsize=(4.5, 4),
                dpi=self.dpi,
            )

            heatmap = COLORMAP[np.uint8(255.0 * cams[i])]
            superimpose = 0.4 * heatmap + images[i][..., np.newaxis]

            axes[0].imshow(normalize(superimpose), aspect="equal")
            set_xticks(
                axis=axes[0], ticks_loc=np.linspace(0, superimpose.shape[1] - 1, 5)
            )
            set_yticks(
                axis=axes[0], ticks_loc=np.linspace(0, superimpose.shape[0] - 1, 5)
            )
            if titles:
                axes[0].set_title(titles[i])

            figure.colorbar(cm.ScalarMappable(cmap=JET), cax=axes[1])

            self.figure(tag=f"{tag}/{i:02d}", figure=figure, step=step, mode=mode)
