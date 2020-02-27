# -*- coding: utf-8 -*-

from antispoofing.mcnns.datasets.livdetiris17_nd import LivDetIris17_ND
from antispoofing.mcnns.datasets.livdetiris17_warsaw import LivDetIris17_Warsaw
from antispoofing.mcnns.datasets.livdetiris17_clarkson import LivDetIris17_Clarkson
from antispoofing.mcnns.datasets.livdetiris17_iiitwvu import LivDetIris17_IIITWVU
from antispoofing.mcnns.datasets.livdetiris17_combined import LivDetIris17_Combined
from antispoofing.mcnns.datasets.livdetiris17_nd_ww import LivDetIris17_ND_WW
from antispoofing.mcnns.datasets.livdetiris17_nd_ww_cl import LivDetIris17_ND_WW_CL
from antispoofing.mcnns.datasets.ndcld15 import  NDCLD15
from antispoofing.mcnns.datasets.forannotation import ForAnnotation
from antispoofing.mcnns.datasets.ndcontactlenses import NDContactLenses


registered_datasets = {0: LivDetIris17_ND,
                       1: LivDetIris17_Warsaw,
                       2: LivDetIris17_ND_WW,
                       3: LivDetIris17_Clarkson,
                       4: LivDetIris17_ND_WW_CL,
                       5: LivDetIris17_IIITWVU,
                       6: LivDetIris17_Combined,
                       7: NDCLD15,
                       8: ForAnnotation,
                       9: NDContactLenses,
                       }
