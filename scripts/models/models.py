#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:50:11 2019

@author: manoj
"""

import os
import sys
# sys.path.append(os.getcwd())
from models import *
from models.densenet_og import DenseNet2D
from models.mobilenet_v1 import MobileNet2D_V1
from models.mobilenet_v2 import MobileNet2D_V2
from models.mobilenet_v3 import MobileNet2D_V3
from models.mobilenet_v4 import MobileNet2D_V4
from models.mobilenet_v5 import MobileNet2D_V5
from models.mobilenet_v1_AP import MobileNet2D_V1_AP
from models.densenet_og_AP import DenseNet2D_AP

model_dict = {}

model_dict['densenet_og'] = DenseNet2D(dropout=True,prob=0.2)

model_dict['mobilenet_v1'] = MobileNet2D_V1(dropout=True, prob=0.2)

model_dict['mobilenet_v2'] = MobileNet2D_V2(dropout=True, prob=0.2)

model_dict['mobilenet_v3'] = MobileNet2D_V3(dropout=True, prob=0.2)

model_dict['mobilenet_v4'] = MobileNet2D_V4(dropout=True, prob=0.2)

model_dict['mobilenet_v5'] = MobileNet2D_V5(dropout=True, prob=0.2)

model_dict['mobilenet_v1_ap'] = MobileNet2D_V1_AP(dropout=True, prob=0.2)

model_dict['densenet_og_ap'] = DenseNet2D_AP(dropout=True, prob=0.2)