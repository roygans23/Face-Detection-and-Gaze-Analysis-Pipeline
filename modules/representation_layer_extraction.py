import torch
from torch import nn

class RepresentationLayerExtraction(object):
    def __init__(self, model: nn.Module, target_layer_name: str):
        super().__init__()

        self._model = model
        self.activation_output = None # This will store the output of the target layer

        # Register the hook on the specified layer
        self.register_hook(target_layer_name)

    def register_hook(self, target_layer_name: str):
        # Find the target layer and register a forward hook
        target_layer = dict(self._model.named_modules()).get(target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in the model.")

        # Define a hook function that saves the output of the target layer
        def hook_fn(module, input, output):
            self.activation_output = output  # Save the output of this layer
            
        target_layer.register_forward_hook(hook_fn)

    def extract_representation(self, input_image: torch.Tensor) -> torch.Tensor:

        # Transform to evaluation mode for predicting input output without gradient calculations
        self._model.eval()
        with torch.no_grad():
            
            # Forward pass to trigger the hook
            _ = self._model(input_image)

        return self.activation_output