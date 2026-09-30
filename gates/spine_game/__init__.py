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

"""A mock bank's days, played against a real hub - the acceptance test, and the demo.

Four modules. `roles` is the bank: three desks and the back office seated at the nodes of one book
by its admin through the CLI, and the acts each seat performs through the MCP binding against a
service on an ephemeral port. `red` is what an adversary tries and what the record answers.
`faults` is what goes wrong with the machinery rather than with anybody's intent. `play` founds
the bank, stands the hub and its followers up, plays the days, writes the SCRIPT of what was asked
beside the run with what the engine answered, and holds the record against it with
`derivus_spine.oracle`.

Nothing is monkeypatched: real homes under the caller's own directory, a real service over a real
socket, real replicas pulling real frames over it, and every fault injected as data on a disk or by
killing a process - the hub, served by `faults` as a process of its own and killed, being the one
besides the harness's own. The hub is the single writer and every seat reaches it the way a model
would.
"""
