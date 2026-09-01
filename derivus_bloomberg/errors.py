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


class BloombergFXError(Exception):
    """Base class for Bloomberg FX adapter failures."""


class BloombergUnavailable(BloombergFXError):
    pass


class BloombergRequestError(BloombergFXError):
    pass


class BloombergEntitlementError(BloombergRequestError):
    pass


class IncompleteSurface(BloombergFXError):
    pass


class DuplicateSurfacePoint(BloombergFXError):
    pass


class UnsupportedFXConvention(BloombergFXError):
    pass


class SurfaceStructureChanged(BloombergFXError):
    pass


class InvalidQuote(BloombergFXError):
    pass


class SurfaceAlreadyInstalled(BloombergFXError):
    pass


class SurfaceNotInstalled(BloombergFXError):
    pass


class BloombergConfigurationError(BloombergFXError):
    pass


# Every adapter's refusals hang off `BloombergFXError`, whose name reads narrower than its scope:
# `derivus.service` and `derivus_mcp` catch it, so one base and one taxonomy rather than two.
class IncompleteChain(BloombergFXError):
    """The listed chain does not carry the ladder that was asked of it - too few distinct contracts
    survived snapping, or an expiry the emitter needs has no admissible print."""


class UnsupportedExerciseStyle(BloombergFXError):
    """The chain's exercise style is not the one the fit assumes. An AMERICAN premium is not the
    European premium a Heston-Nandi calibration prices against, so it refuses rather than fits."""


class IncompleteStrip(BloombergFXError):
    """The verified strip does not carry the curve that was asked of it - too few points survived
    screening, or two benchmarks land on one knot, which the bootstrap cannot identify apart."""


class IncompleteLadder(BloombergFXError):
    """The swaption grid does not carry the ladder that was asked of it - too few cells survived
    screening for the fitted parameters to be identified by what the terminal served."""


def raise_response_error(text: str) -> None:
    """Raise Bloomberg's own refusal text under the type it warrants: `BloombergEntitlementError`
    when the message names an authorisation problem, `BloombergRequestError` otherwise."""
    error_type = BloombergEntitlementError if any(
        token in text.upper() for token in ('NOT_AUTHORIZED', 'NOT_ENTITLED', 'NO_AUTH')) \
        else BloombergRequestError
    raise error_type(text)