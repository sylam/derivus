########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import pandas as pd

QuoteType = Literal['ATM', 'RR', 'BF']
QuoteCoordinate = tuple[str, QuoteType, float | None]


@dataclass(frozen=True)
class FXQuoteSecurity:
    security: str
    value_field: str = 'PX_LAST'


@dataclass(frozen=True)
class FXVolDefinition:
    pair: str
    surface_name: str
    currency: str
    expiries: Mapping[str, float]
    pillars: tuple[float, ...]
    securities: Mapping[QuoteCoordinate, FXQuoteSecurity]
    quote_scale: float = 0.01
    delta_type: str = 'Forward'
    premium_adjusted: bool = True
    atm_convention: str = 'Delta_Neutral_Straddle'
    grid_tolerance: float = 1e-4
    quote_sensitivity: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'expiries', MappingProxyType(dict(self.expiries)))
        object.__setattr__(self, 'pillars', tuple(self.pillars))
        object.__setattr__(self, 'securities', MappingProxyType(dict(self.securities)))


@dataclass(frozen=True)
class RawBloombergObservation:
    expiry_label: str
    quote_type: QuoteType
    pillar: float | None
    security: str
    field: str
    value: object
    #: the two-way the terminal quoted beside the value, raw and OPTIONAL - a pillar answered with
    #: no PX_BID/PX_ASK arrives here as None and stays mid-only all the way to the block
    bid: object = None
    ask: object = None


@dataclass(frozen=True)
class FXVolPoint:
    expiry_label: str
    expiry: float
    quote_type: QuoteType
    pillar: float | None
    value: float
    observed_at: pd.Timestamp
    security: str
    field: str
    raw_value: float
    #: the two-way in the surface's own units, defaulted absent. `value` is the mid the surface,
    #: the bootstrap and every mark are built from; these two are carried BESIDE it for the quote
    #: layer, which is the only thing in the house that opens them
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class FXVolSnapshot:
    pair: str
    surface_name: str
    currency: str
    points: tuple[FXVolPoint, ...]
    retrieved_at: pd.Timestamp
    delta_type: str
    premium_adjusted: bool
    atm_convention: str
    grid_tolerance: float
    quote_sensitivity: bool