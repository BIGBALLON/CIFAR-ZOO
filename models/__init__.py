from .alexnet import *
from .vgg import *
from .resnet import *
from .preresnet import *
from .senet import *
from .resnext import *
from .densenet import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)
