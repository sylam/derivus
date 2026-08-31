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


# The two refusals the listed EQUITY chain adds. They hang off `BloombergFXError` with everything
# else: the base is named for the adapter's first market rather than for its scope, and renaming it
# would reach across into `derivus.service` and `derivus_mcp`, which catch it. One taxonomy, one
# base, and the name stays historical rather than becoming two.
class IncompleteChain(BloombergFXError):
    """The listed chain does not carry the ladder that was asked of it - too few distinct contracts
    survived snapping, or an expiry the emitter needs has no admissible print."""


class UnsupportedExerciseStyle(BloombergFXError):
    """The chain's exercise style is not the one the fit assumes. An AMERICAN premium is not the
    European premium a Heston-Nandi calibration prices against, so it refuses rather than fits."""


def raise_response_error(text: str) -> None:
    """Bloomberg's own refusal text, typed and raised - an entitlement problem is a thing to go
    and fix, anything else is a request that failed.

    It lives here rather than in the reader because two callers now make the same refusal: the
    session's strict reader, and `fxvol`, which reads the value out of the TOLERANT reader and
    applies the strict policy to it itself. One wording, one typing, whichever side spots it.
    """
    error_type = BloombergEntitlementError if any(
        token in text.upper() for token in ('NOT_AUTHORIZED', 'NOT_ENTITLED', 'NO_AUTH')) \
        else BloombergRequestError
    raise error_type(text)