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