# -*- coding: utf-8 -*-

from antispoofing.mcnns.metaclassification.metasvm import MetaSVM
from antispoofing.mcnns.metaclassification.metarandomforest import MetaRandomForest


meta_ml_algo = {0: MetaSVM,
                1: MetaRandomForest,
                }