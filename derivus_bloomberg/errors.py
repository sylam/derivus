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