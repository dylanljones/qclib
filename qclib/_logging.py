# coding: utf-8
#
# This code is part of qclib.
# 
# Copyright (c) 2021, Dylan Jones

import logging

# Create logger
logger = logging.getLogger("qclib")
logger.setLevel(logging.WARNING)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
