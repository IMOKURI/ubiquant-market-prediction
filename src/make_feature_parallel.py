import logging
import os
from typing import List

import pandas as pd

from .feature_store import Store
from .features.base import Context, get_feature, get_feature_schema, get_features, normalize_feature_name

log = logging.getLogger(__name__)


def _make_feature(c, feature_name: str, index: int, in_total: int):
    pass
