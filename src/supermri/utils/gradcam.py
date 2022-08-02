import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from torchvision import transforms


class LayerHook:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients, self.activations = None, None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.cpu().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].cpu().detach()

    def __call__(self, x):
        self.gradients, self.activations = None, None
        return self.model(x)


class GradCAM:
    def __init__(self, args, model):
        self.model = deepcopy(model.eval())
        self.model_hook = LayerHook(self.model, self.model.activation)
        self.resize = transforms.Resize(size=args.input_shape[1:])

        self.loss_function = F.binary_cross_entropy

    def resize_and_normalize(self, x):
        """resize x to input_shape and normalize to [0, 1]"""
        x = self.resize(x)
        x_flat = x.view(-1, x.shape[1] * x.shape[2])
        x_min = x_flat.min(dim=1)[0][:, None, None]
        x_max = x_flat.max(dim=1)[0][:, None, None]
        x = (x - x_min) / (x_max - x_min)
        return x

    def forward(self, inputs, is_real: bool):
        outputs = self.model_hook(inputs)
        outputs = F.sigmoid(outputs)

        self.model.zero_grad()
        labels = torch.ones_like(outputs) if is_real else torch.zeros_like(outputs)
        loss = self.loss_function(outputs, labels)
        loss.backward(retain_graph=True)

        activations = self.model_hook.activations
        gradients = self.model_hook.gradients

        weights = torch.mean(gradients, dim=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations
        cam = torch.sum(weighted_activations, dim=1)
        cam = torch.maximum(cam, torch.zeros_like(cam))

        cam = self.resize_and_normalize(cam)

        return torch.squeeze(outputs, dim=1).cpu().detach(), cam
