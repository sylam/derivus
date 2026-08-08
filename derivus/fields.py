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

"""DEPRECATED for one release - import from `derivus.schema` instead.

`fields.mapping` was the documented surface and derivus is published on PyPI, so the two names an
external caller could plausibly have bound stay importable from here. There is nothing else left:
every store is emitted from the per-class `fields` declarations, and `schema.py` is where the
vocabulary, the emitters and the assembly now all live.
"""

from .schema import default, mapping                                                  # noqa: F401
