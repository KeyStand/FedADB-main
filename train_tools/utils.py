from .models import *

__all__ = ["create_models"]

MODELS = {
    "res18": resnet.resnet18,
}


NUM_CLASSES = {
    "cifar10": 10,
}

IMG_SIZES = {
    "cifar10": 32,
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)

    return model
