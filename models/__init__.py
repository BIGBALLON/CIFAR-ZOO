# -*-coding:utf-8-*-
from .lenet import *
from .alexnet import *
from .vgg import *
from .resnet import *
from .preresnet import *
from .senet import *
from .resnext import *
from .densenet import *
from .shake_shake import *
from .sknet import *
from .genet import *
from .cbam_resnext import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)
