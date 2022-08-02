import torch
import numpy as np
import typing as t

from supermri.utils import utils


class EarlyStopping:
    """Early Stopping module
    The monitor function monitors the validation loss and compare it against the
    previous best loss recorded. After min_epochs number of epochs, if the
    current loss value has not improved for more than patience number of epoch,
    then send terminate flag and load the best weight for the model.
    """

    def __init__(
        self,
        args,
        model: torch.nn.Module,
        patience: int = 10,
        min_epochs: int = 50,
    ):
        self.args = args
        self.model = model
        self.patience = patience
        self.min_epochs = min_epochs
        self.device = args.device

        self.wait = 0
        self.best_loss = np.inf
        self.best_epoch = -1

        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint = self.checkpoint_dir / "best_model.pt"

    def monitor(self, loss: t.Union[float, torch.Tensor], epoch: int):
        terminate = False
        if torch.is_tensor(loss):
            loss = loss.item()
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.wait = 0
            utils.save_model(model=self.model, epoch=epoch, filename=self.checkpoint)
        elif epoch > self.min_epochs:
            if self.wait < self.patience:
                self.wait += 1
            else:
                terminate = True
                self.restore()
                if self.args.verbose:
                    print(
                        f"EarlyStopping: model has not improved in {self.wait} epochs.\n"
                    )
        return terminate

    def restore(self):
        """Restore the best weights to the model"""
        utils.load_checkpoint(self.args, model=self.model, filename=self.checkpoint)
