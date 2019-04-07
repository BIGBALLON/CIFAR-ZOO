from .resnet import *
from .preresnet import *
from .alexnet import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)
