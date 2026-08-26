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

import math
from collections import Counter
from typing import Iterable, Mapping, Protocol

import pandas as pd

from .errors import (BloombergConfigurationError, DuplicateSurfacePoint, IncompleteSurface,
                     InvalidQuote, SurfaceAlreadyInstalled, SurfaceNotInstalled,
                     SurfaceStructureChanged, UnsupportedFXConvention)
from .types import (FXQuoteSecurity, FXVolDefinition, FXVolPoint, FXVolSnapshot, QuoteCoordinate,
                    RawBloombergObservation)

SUPPORTED_CONVENTION = ('Forward', True, 'Delta_Neutral_Straddle')
BOOTSTRAPPER = 'FXVolSurfaceParameters'


class ReferenceDataSource(Protocol):
    def reference_data(self, securities: list[str], fields: list[str]) -> Mapping[str, Mapping[str, object]]:
        ...


def _expected_coordinates(definition: FXVolDefinition) -> set[QuoteCoordinate]:
    return {(expiry, 'ATM', None) for expiry in definition.expiries} | {
        (expiry, quote_type, pillar)
        for expiry in definition.expiries
        for pillar in definition.pillars
        for quote_type in ('RR', 'BF')}


def validate_definition(definition: FXVolDefinition) -> None:
    convention = (definition.delta_type, definition.premium_adjusted, definition.atm_convention)
    if convention != SUPPORTED_CONVENTION:
        raise UnsupportedFXConvention(
            '{} / premium_adjusted={} / {} is unsupported; expected {} / '
            'premium_adjusted={} / {}'.format(*convention, *SUPPORTED_CONVENTION))
    if not definition.expiries:
        raise IncompleteSurface('the surface has no expiries')
    if len(set(definition.expiries.values())) != len(definition.expiries):
        raise BloombergConfigurationError('expiry year fractions must be unique')
    if any(not math.isfinite(value) or value <= 0.0 for value in definition.expiries.values()):
        raise BloombergConfigurationError('expiry year fractions must be positive and finite')
    if len(set(definition.pillars)) != len(definition.pillars):
        raise BloombergConfigurationError('delta pillars must be unique')
    if any(not math.isfinite(pillar) or not 0.0 < pillar < 0.5 for pillar in definition.pillars):
        raise BloombergConfigurationError('delta pillars must be finite and strictly between 0 and 0.5')
    if not math.isfinite(definition.quote_scale) or definition.quote_scale == 0.0:
        raise BloombergConfigurationError('quote_scale must be finite and non-zero')

    expected = _expected_coordinates(definition)
    supplied = set(definition.securities)
    if supplied != expected:
        missing = sorted(expected - supplied, key=str)
        extra = sorted(supplied - expected, key=str)
        raise IncompleteSurface('security map differs from configured surface; missing={}, extra={}'.format(
            missing, extra))
    for coordinate, quote in definition.securities.items():
        if not isinstance(quote, FXQuoteSecurity) or not quote.security or not quote.value_field:
            raise BloombergConfigurationError('invalid Bloomberg security mapping for {}'.format(coordinate))


def normalize_fx_vol(definition: FXVolDefinition,
                     observations: Iterable[RawBloombergObservation],
                     retrieved_at: pd.Timestamp) -> FXVolSnapshot:
    validate_definition(definition)
    retrieved_at = pd.Timestamp(retrieved_at)
    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.tz_localize('UTC')
    else:
        retrieved_at = retrieved_at.tz_convert('UTC')

    observations = tuple(observations)
    coordinates = [(item.expiry_label, item.quote_type, item.pillar) for item in observations]
    duplicates = sorted((coordinate for coordinate, count in Counter(coordinates).items() if count > 1), key=str)
    if duplicates:
        raise DuplicateSurfacePoint('duplicate Bloomberg observations for {}'.format(duplicates))

    expected = _expected_coordinates(definition)
    actual = set(coordinates)
    if actual != expected:
        raise IncompleteSurface('Bloomberg response differs from configured surface; missing={}, extra={}'.format(
            sorted(expected - actual, key=str), sorted(actual - expected, key=str)))

    points = []
    for observation in observations:
        coordinate = (observation.expiry_label, observation.quote_type, observation.pillar)
        configured = definition.securities[coordinate]
        if (observation.security, observation.field) != (configured.security, configured.value_field):
            raise BloombergConfigurationError('observation source does not match mapping for {}'.format(coordinate))
        try:
            raw_value = float(observation.value)
        except (TypeError, ValueError) as error:
            raise InvalidQuote('{} returned {!r}'.format(coordinate, observation.value)) from error
        value = raw_value * definition.quote_scale
        if not math.isfinite(value) or (observation.quote_type == 'ATM' and value <= 0.0):
            raise InvalidQuote('{} returned invalid quote {!r}'.format(coordinate, observation.value))
        points.append(FXVolPoint(
            observation.expiry_label, definition.expiries[observation.expiry_label],
            observation.quote_type, observation.pillar, value, retrieved_at,
            observation.security, observation.field, raw_value))

    quote_order = {'RR': 0, 'BF': 1}
    points.sort(key=lambda point: (
        point.expiry, point.quote_type != 'ATM',
        point.pillar if point.pillar is not None else -1.0,
        quote_order.get(point.quote_type, -1)))
    return FXVolSnapshot(
        definition.pair, definition.surface_name, definition.currency, tuple(points), retrieved_at,
        definition.delta_type, definition.premium_adjusted, definition.atm_convention,
        definition.grid_tolerance, definition.quote_sensitivity)


def fetch_fx_vol(source: ReferenceDataSource, definition: FXVolDefinition) -> FXVolSnapshot:
    validate_definition(definition)
    securities_by_field = {}
    for quote in definition.securities.values():
        securities_by_field.setdefault(quote.value_field, set()).add(quote.security)
    response = {}
    for field in sorted(securities_by_field):
        field_response = source.reference_data(sorted(securities_by_field[field]), [field])
        for security, values in field_response.items():
            response.setdefault(security, {}).update(values)
    retrieved_at = pd.Timestamp.now(tz='UTC')

    observations = []
    for (expiry, quote_type, pillar), quote in definition.securities.items():
        try:
            value = response[quote.security][quote.value_field]
        except KeyError as error:
            raise IncompleteSurface('{} {} is missing from the Bloomberg response'.format(
                quote.security, quote.value_field)) from error
        observations.append(RawBloombergObservation(
            expiry, quote_type, pillar, quote.security, quote.value_field, value))
    return normalize_fx_vol(definition, observations, retrieved_at)


def validate_snapshot(snapshot: FXVolSnapshot) -> None:
    convention = (snapshot.delta_type, snapshot.premium_adjusted, snapshot.atm_convention)
    if convention != SUPPORTED_CONVENTION:
        raise UnsupportedFXConvention(
            '{} / premium_adjusted={} / {} is unsupported'.format(*convention))
    if not snapshot.points:
        raise IncompleteSurface('the snapshot has no points')
    if not 1e-8 <= snapshot.grid_tolerance <= 1.0:
        raise InvalidQuote('Grid_Tolerance must be between 1e-8 and 1.0')

    coordinates = [(point.expiry, point.quote_type, point.pillar) for point in snapshot.points]
    duplicates = sorted((coordinate for coordinate, count in Counter(coordinates).items()
                         if count > 1), key=str)
    if duplicates:
        raise DuplicateSurfacePoint('duplicate snapshot points for {}'.format(duplicates))
    expiry_by_label = {}
    label_by_expiry = {}
    for point in snapshot.points:
        if (point.expiry_label in expiry_by_label and
                expiry_by_label[point.expiry_label] != point.expiry):
            raise InvalidQuote('each expiry label must map to exactly one year fraction')
        if point.expiry in label_by_expiry and label_by_expiry[point.expiry] != point.expiry_label:
            raise InvalidQuote('each expiry year fraction must map to exactly one label')
        expiry_by_label[point.expiry_label] = point.expiry
        label_by_expiry[point.expiry] = point.expiry_label

    expiries = {point.expiry for point in snapshot.points}
    pillars = {point.pillar for point in snapshot.points if point.quote_type != 'ATM'}
    if any(pillar is None or not 0.0 < pillar < 0.5 for pillar in pillars):
        raise InvalidQuote('wing pillars must be strictly between 0 and 0.5')
    expected = {(expiry, 'ATM', None) for expiry in expiries} | {
        (expiry, quote_type, pillar)
        for expiry in expiries for pillar in pillars for quote_type in ('RR', 'BF')}
    if set(coordinates) != expected:
        raise IncompleteSurface('snapshot point set is incomplete')

    retrieved_at = pd.Timestamp(snapshot.retrieved_at)
    for point in snapshot.points:
        if not math.isfinite(point.value) or (point.quote_type == 'ATM' and point.value <= 0.0):
            raise InvalidQuote('{} {} has invalid value {!r}'.format(
                point.expiry_label, point.quote_type, point.value))
        if pd.Timestamp(point.observed_at) != retrieved_at:
            raise InvalidQuote('every point must carry the snapshot retrieval timestamp')


def to_market_prices_block(snapshot: FXVolSnapshot) -> dict:
    validate_snapshot(snapshot)
    points = [{
        'Use': 'Yes',
        'Expiry': point.expiry,
        'Pillar': 0.0 if point.pillar is None else point.pillar,
        'Quote_Type': point.quote_type,
        'Quoted_Market_Value': point.value,
        'Timestamp': point.observed_at,
    } for point in snapshot.points]
    return {'instrument': {
        'Currency': snapshot.currency,
        'Delta_Type': snapshot.delta_type,
        'Premium_Adjusted': 'Yes' if snapshot.premium_adjusted else 'No',
        'ATM_Convention': snapshot.atm_convention,
        'Grid_Tolerance': snapshot.grid_tolerance,
        'Quote_Sensitivity': 'Yes' if snapshot.quote_sensitivity else 'No',
        'Points': points,
    }}


def _market_price_name(snapshot: FXVolSnapshot) -> str:
    return 'FXVolPrices.{}'.format(snapshot.surface_name)


def _market_prices(config) -> dict:
    try:
        params = config.params
        if BOOTSTRAPPER not in params['Bootstrapper Configuration']:
            raise BloombergConfigurationError(
                'Bootstrapper Configuration has no {} entry'.format(BOOTSTRAPPER))
        return params['Market Prices']
    except (AttributeError, KeyError) as error:
        raise BloombergConfigurationError('expected a derivus Config with market-price stores') from error


def _structure(block: dict) -> tuple:
    instrument = block['instrument']
    points = tuple((point['Use'], point['Expiry'], point['Pillar'], point['Quote_Type'])
                   for point in instrument['Points'])
    return (instrument['Currency'], instrument['Delta_Type'], instrument['Premium_Adjusted'],
            instrument['ATM_Convention'], instrument['Grid_Tolerance'],
            instrument['Quote_Sensitivity'], points)


def install_fx_vol_snapshot(config, snapshot: FXVolSnapshot) -> str:
    market_prices = _market_prices(config)
    name = _market_price_name(snapshot)
    if name in market_prices:
        raise SurfaceAlreadyInstalled('{} already exists'.format(name))
    replacement = to_market_prices_block(snapshot)
    market_prices[name] = replacement
    return name


def update_fx_vol_snapshot(config, snapshot: FXVolSnapshot) -> str:
    market_prices = _market_prices(config)
    name = _market_price_name(snapshot)
    if name not in market_prices:
        raise SurfaceNotInstalled('{} is not installed'.format(name))
    replacement = to_market_prices_block(snapshot)
    if _structure(market_prices[name]) != _structure(replacement):
        raise SurfaceStructureChanged('{} structure differs from the installed surface'.format(name))
    market_prices[name] = replacement
    return name