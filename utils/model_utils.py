import torch
import torch.nn as nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1

def build_base_model(arch, num_classes, pretrained, train_mode):
    if arch == 'vgg':
        model = models.vgg16(pretrained=pretrained)
        if train_mode:
            override_classifier_head(model, num_classes, arch)
    elif arch == 'resnet':
        pretrained_type = 'vggface2' if pretrained else False
        model = InceptionResnetV1(pretrained=pretrained_type, classify=train_mode, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

def override_classifier_head(model, num_classes, arch):
    if arch == 'vgg':
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == 'resnet':
        in_features = model.logits.in_features
        model.logits = nn.Linear(in_features, num_classes)

def adjust_classifier_for_eval(model, state_dict, arch):
    if arch == 'vgg' and 'classifier.6.weight' in state_dict:
        out_features = state_dict['classifier.6.weight'].shape[0]
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, out_features)
    elif arch == 'resnet' and 'logits.weight' in state_dict:
        out_features = state_dict['logits.weight'].shape[0]
        model.logits = nn.Linear(model.logits.in_features, out_features)

def remove_classifier_head(state_dict, arch):
    if arch == 'vgg':
        return {k: v for k, v in state_dict.items() if not k.startswith('classifier.6')}
    elif arch == 'resnet':
        return {k: v for k, v in state_dict.items() if not k.startswith('logits')}


def freeze_base_layer_grads(model, arch, additional_keywords=None):
    classifier = 'classifier.6' if arch == 'vgg' else 'logits'
    trainable = [classifier] + (additional_keywords or [])
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in trainable)

def display_params_grad_calc_status(model):
    for name, param in model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

import torch.nn as nn

def print_conv_named_parameters(model):
    print("Convolutional Layers:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            for param_name, _ in module.named_parameters(recurse=False):
                print(f"{name}.{param_name}")

def display_checkpoint_metadata(checkpoint):
    for k in ['acc', 'epoch']:
        if k in checkpoint:
            print(f"{k}: {checkpoint[k]}")
    if 'optimizer' in checkpoint:
        print(f"Optimizer state: {checkpoint['optimizer']['param_groups']}")

def copy_batchnorm_stats(source_model, target_model):
    for (name1, mod1), (name2, mod2) in zip(source_model.named_modules(), target_model.named_modules()):
        if isinstance(mod1, nn.BatchNorm2d):
            mod2.running_mean.data.copy_(mod1.running_mean.data)
            mod2.running_var.data.copy_(mod1.running_var.data)
