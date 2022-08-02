import torch
import torchinfo
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from supermri.utils import utils

_CRITICS = dict()


def register(name):
    """Note: update __init__.py for additional models"""

    def add_to_dict(fn):
        global _CRITICS
        _CRITICS[name] = fn
        return fn

    return add_to_dict


def uniform(inputs: torch.Tensor, minval: float = 0.0, maxval: float = 1.0):
    """
    Returns a tensor with the same size as inputs that is filled with random
    numbers from a uniform distribution in range [minval, maxval)
    """
    return (minval - maxval) * torch.rand_like(inputs) + maxval


class Critic:
    """Adversarial loss for targets and up-sampled images"""

    def __init__(self, args, summary=None):
        assert args.critic in _CRITICS, f"Critic {args.critic} not found."

        self.name = args.critic
        self.critic_steps = args.critic_steps
        self.label_smoothing = args.label_smoothing
        self.mixed_precision = args.mixed_precision

        self.model = _CRITICS[args.critic](args)
        self.model.to(args.device)

        self.output_shape = self.model(
            torch.rand(2, *args.input_shape, device=args.device)
        ).shape[1:]

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.critic_lr, betas=(0.5, 0.999)
        )
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.loss_function = F.binary_cross_entropy_with_logits

        # get critic model  summary and write to args.output_dir
        model_info = torchinfo.summary(
            self.model,
            input_size=(args.batch_size, *args.input_shape),
            device=args.device,
            verbose=0,
        )
        with open(args.output_dir / "critic.txt", "w") as file:
            file.write(str(model_info))
        if args.verbose == 2:
            print(str(model_info))
        if summary is not None:
            summary.scalar("critic/trainable_parameters", model_info.trainable_params)

    def real_labels(self, inputs: torch.Tensor):
        if self.label_smoothing:
            labels = uniform(inputs, minval=0.9, maxval=1.0)
        else:
            labels = torch.ones_like(inputs)
        return labels

    def fake_labels(self, inputs: torch.Tensor):
        if self.label_smoothing:
            labels = uniform(inputs, minval=0.0, maxval=0.1)
        else:
            labels = torch.zeros_like(inputs)
        return labels

    def critic_loss(
        self, discriminate_real: torch.Tensor, discriminate_fake: torch.Tensor
    ):
        real_loss = self.loss_function(
            discriminate_real, self.real_labels(discriminate_real)
        )
        fake_loss = self.loss_function(
            discriminate_fake, self.fake_labels(discriminate_fake)
        )
        return (real_loss + fake_loss) / 2

    def train(self, real: torch.Tensor, fake: torch.Tensor) -> dict:
        """Train critic model on real and fake samples for self.critic_steps
        Args:
          real: real samples
          fake: fake samples
        Returns:
          results: dictionary containing the average loss and the average outputs
                  of the critic on real and fake samples
        """
        assert real.shape == fake.shape
        self.model.train()

        results = {}
        for _ in range(self.critic_steps):
            self.optimizer.zero_grad()
            with autocast(enabled=self.mixed_precision):
                discriminate_real = self.model(real)
                discriminate_fake = self.model(fake)

                loss = self.critic_loss(discriminate_real, discriminate_fake)

                discriminate_real = F.sigmoid(discriminate_real)
                discriminate_fake = F.sigmoid(discriminate_fake)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            utils.update_dict(
                results,
                {
                    "critic/loss": loss.detach().clone(),
                    "critic/D(real)": discriminate_real.mean().detach().clone(),
                    "critic/D(fake)": discriminate_fake.mean().detach().clone(),
                },
            )

        return {k: torch.stack(v).mean() for k, v in results.items()}

    def validate(self, real: torch.Tensor, fake: torch.Tensor) -> dict:
        """Validate critic model on real and fake samples
        Args:
          real: real samples
          fake: fake samples
        Returns:
          results: dictionary containing the average loss and the average outputs
                  of the critic on real and fake samples
        """
        assert real.shape == fake.shape
        self.model.eval()

        with autocast(enabled=self.mixed_precision), torch.no_grad():
            discriminate_real = self.model(real)
            discriminate_fake = self.model(fake)

            loss = self.critic_loss(discriminate_real, discriminate_fake)

            discriminate_real = F.sigmoid(discriminate_real)
            discriminate_fake = F.sigmoid(discriminate_fake)

        return {
            "critic/loss": loss.clone(),
            "critic/D(real)": discriminate_real.mean().clone(),
            "critic/D(fake)": discriminate_fake.mean().clone(),
        }

    def predict(self, x: torch.Tensor, return_mean: bool = True) -> torch.Tensor:
        """Discriminate sample x
        Predictions is close to 1 if x resemble real samples.

        Args:
          x: sample to discriminate
          return_mean: return average of the output
        Returns:
          score: critic output score
        """
        with autocast(enabled=self.mixed_precision):
            score = F.sigmoid(self.model(x))
        return score.mean() if return_mean else score
