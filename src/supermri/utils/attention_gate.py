import torch
import typing as t
from copy import deepcopy
import torch.nn.functional as F


class AGHook:
    """
    Helper class to record the inputs and sigmoid activation for each attention
    gate in the model
    """

    def __init__(self, model, output_logits: bool):
        self.model = deepcopy(model)
        self.output_logits = output_logits
        self.gate_inputs = {}
        self.gate_masks = {}

        # register hook to store attention gate inputs and sigmoid mask
        num_gates = len(model.up_blocks)
        for i in range(num_gates):
            self.model.up_blocks[i].attention_gate.gating_conv.register_forward_hook(
                self._save_gate_input(name=f"gate_{i}")
            )
            self.model.up_blocks[i].attention_gate.sigmoid.register_forward_hook(
                self._save_gate_mask(name=f"gate_{i}")
            )

    def _save_gate_input(self, name: str):
        def hook(module, input, output):
            self.gate_inputs[name] = input[0].cpu().detach()

        return hook

    def _save_gate_mask(self, name: str):
        def hook(module, input, output):
            self.gate_masks[name] = output.cpu().detach()

        return hook

    def __call__(self, x):
        self.gate_inputs, self.gate_masks = {}, {}
        with torch.no_grad():
            outputs = self.model(x)
            if self.output_logits:
                outputs = F.sigmoid(outputs)
        return outputs
