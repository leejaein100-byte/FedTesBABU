# -*- coding: utf-8 -*-
from utils.misc import *
import settings_CUB
# Import the push module containing push_prototypes_top3
import push  # Make sure this import points to your push module

args = settings_CUB.args
model_path = "/home/jilee/jilee/FedTesBABU/saved_models/Stanford_dog/resnet50/False/2025-10-29_11-40-23/FedTesBABU/final_model.pth"
load_model_and_push(model_path, args)