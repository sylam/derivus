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

"""A day on the desk, played against a real hub - the acceptance test, and the demo.

Four modules. `roles` is the desk: seven seats, the capabilities document that scopes them, and the
acts each performs through the MCP binding against a service on an ephemeral port. `red` is what an
adversary tries and what the record answers. `faults` is what goes wrong with the machinery rather
than with anybody's intent. `play` stands the hub and its followers up, plays the day, writes the
SCRIPT of what was asked beside the run, and holds the record against it with
`derivus_spine.oracle`.

Nothing is monkeypatched: real homes under the caller's own directory, a real service over a real
socket, real replicas pulling real frames over it, and every fault injected as data on a disk or by
killing a process - the writer `faults` kills being the only process here besides the harness's own.
The hub is the single writer and every seat reaches it the way a model would.
"""
