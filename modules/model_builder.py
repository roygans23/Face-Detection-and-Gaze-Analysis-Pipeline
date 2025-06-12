from config.model_config import ModelConfig
from utils.model_utils import *
import torch

class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    def initialize_model(self):
        print("Initializing model...")
        model = build_base_model(
            self.config.arch,
            self.config.num_classes,
            self.config.pretrained,
            self.config.train_mode
        )

        if self.config.checkpoint_path:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.config.device)

            if self.config.data_parallel_patch:
                model.features = torch.nn.DataParallel(model.features)

            state_dict = checkpoint.get('state_dict', checkpoint)

            if self.config.train_mode:
                state_dict = remove_classifier_head(state_dict, self.config.arch)
                model.load_state_dict(state_dict, strict=False)
                override_classifier_head(model, self.config.num_classes, self.config.arch)
            else:
                adjust_classifier_for_eval(model, state_dict, self.config.arch)
                model.load_state_dict(state_dict)

            if self.config.show_logs:
                display_checkpoint_metadata(checkpoint)

        if self.config.freeze_grads and self.config.train_mode:
            freeze_base_layer_grads(model, self.config.arch, self.config.additional_trainable_keywords)

        if self.config.show_logs:
            print_conv_named_parameters(model)
            display_params_grad_calc_status(model)

        self.model = model.to(self.config.device)
        return self.model

    def copy_bn_from_pretrained(self) -> torch.nn.Module:
        print("Copying BatchNorm stats from pretrained model...")

        if self.model is None:
            raise ValueError("Model not initialized yet. Call initialize_model() first.")

        pretrained_config = ModelConfig(
            arch=self.config.arch,
            pretrained=True,
            checkpoint_path=None,
            train_mode=False,
            freeze_grads=True,
            device=self.config.device
        )
        pretrained_model = ModelBuilder(pretrained_config).initialize_model()
        copy_batchnorm_stats(pretrained_model, self.model)

