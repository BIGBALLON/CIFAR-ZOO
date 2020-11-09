# -*-coding:utf-8-*-
from .alexnet import *
from .cbam_resnext import *
from .densenet import *
from .genet import *
from .lenet import *
from .preresnet import *
from .resnet import *
from .resnext import *
from .senet import *
from .shake_shake import *
from .sknet import *
from .vgg import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)
