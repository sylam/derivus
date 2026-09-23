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

"""May this quote be booked - asked in three ways, of which two refuse and one reports.

The PLAN dimension asks about the book: the marginal charge was solved against a portfolio that has
since moved, so the residual this trade would leave is not the one that was priced. That is an
EQUALITY and it refuses.

The PILLAR dimension asks how old the board was WHEN THE QUOTE WAS STRUCK: the age of the oldest
stamped quote row the book carried, against the window a desk declared. A home declaring none has
said nothing about how stale is too stale, and nothing here invents a number for it.

The MARKET is REPORTED and never refused. Between a quote and the client's word the board moves,
which is the ordinary condition of quoting rather than a fault: the desk's own
`Quote Policy.firm_seconds` is the promise that bounds it, and what the booking owes a reader is
which market it was struck on and which one it lands in.

Pure functions over plain data - two pairs of hashes, one age, one window - holding no log, clock or
home, so the same inputs answer the same way on the hub, on a replica and in a gate. `assess`
returns a verdict; `check` raises the same answer as `QuoteNotFirm`.
"""
from .errors import MalformedEvent, QuoteNotFirm
from .policy import PILLAR_SECONDS
from .vocabulary import is_hash, is_number

#: The three things a quote is asked about, named once so a refusal, a verdict and a gate all spell
#: them the same way. The first two refuse; the third is the reading beside them.
PLAN = 'plan'
PILLAR = 'pillar'
MARKET = 'market'

#: The refusal wording for a book that moved under the solve.
MOVED = ('the book moved under it: the charge was solved against plan {pinned} and the book '
         'standing now is {current} - re-quote, because the residual this trade leaves is a '
         'property of the portfolio it joins, and that portfolio is not the one that was priced')
#: The refusal wording for a board that was already stale when the price was given.
AGED = ('the board it was struck on was {age:.1f}s old at its oldest stamped pillar, against the '
        '{window:.0f}s window declared here - re-quote off a ticked market: a pillar older than '
        'the cadence that refreshes it was read off a board nobody is watching')
#: And for a board whose age cannot be established at all: an unknown age is not an age inside a
#: window, and a desk that declared one asked for the question to be answered.
UNKNOWN = ('the board it was struck on carries no stamped pillar, so its age cannot be measured '
           'against the {window:.0f}s window declared here - stamp the quotes the book is built '
           'from, or declare no window')


def assess(pinned, current, pillar_age, policy):
    """The verdict on one quote: `{plan, pillar, market, firm, refusals}`.

    `pinned` and `current` are `{'plan_hash', 'values_hash'}` - what the quote pinned, and what the
    book and its market are right now. `pillar_age` is how old the board's oldest stamped pillar
    was when the quote was struck, in seconds, or None where no row was stamped. `policy` is what
    `policy.firmness_in_force` answers.

    Each answer reports everything it read - the two hashes, the age and the window - so a caller
    shows why a booking refused, or which market it moved through, without re-deriving anything.
    """
    for name, mapping in (('pinned', pinned), ('current', current)):
        if not isinstance(mapping, dict):
            raise MalformedEvent(
                'firmness: {} is {}, not the {{plan_hash, values_hash}} pair a quote pins - the '
                'check compares two pairs and cannot be asked about half of one'.format(
                    name, type(mapping).__name__))
    window, age = _window(policy), _age(pillar_age)
    was, now = _hash(pinned, 'plan_hash', 'the quote pinned'), _hash(current, 'plan_hash',
                                                                    'the book shows')
    struck, board = _hash(pinned, 'values_hash', 'the quote pinned'), _hash(current, 'values_hash',
                                                                           'the book shows')
    stale = None if window is None else (
        UNKNOWN.format(window=window) if age is None
        else AGED.format(age=age, window=window) if age > window else None)
    refusals = [said for said in (MOVED.format(pinned=was, current=now) if was != now else None,
                                  stale) if said]
    return {PLAN: {'pinned': was, 'current': now, 'moved': was != now},
            PILLAR: {'age': age, 'window': window, 'firm': stale is None},
            MARKET: {'pinned': struck, 'current': board, 'moved': struck != board},
            'firm': not refusals, 'refusals': refusals}


def check(pinned, current, pillar_age, policy, quote_id=None):
    """`assess`, raised: the verdict when the quote may be booked, `QuoteNotFirm` when it may not.

    The refusal names every answer that failed, not the first - a moved book and a stale board have
    two remedies, and reporting one sends the caller back into the other.
    """
    verdict = assess(pinned, current, pillar_age, policy)
    if not verdict['firm']:
        raise QuoteNotFirm('quote {} cannot be booked. {}'.format(
            quote_id if quote_id is not None else '(unnamed)', ' AND '.join(verdict['refusals'])))
    return verdict


def _window(policy):
    """The declared pillar window in seconds, or None where this home declared none.

    Absence is the answer rather than a default: a home that has not said how stale is too stale
    refuses nothing on staleness and reports the age it measured.
    """
    if not isinstance(policy, dict):
        raise MalformedEvent(
            'firmness: the policy is {}, not the document `policy.firmness_in_force` answers - '
            'hand in that document, which is the declared {{"{}": seconds}} or the empty object a '
            'home declaring none runs on'.format(type(policy).__name__, PILLAR_SECONDS))
    window = policy.get(PILLAR_SECONDS)
    if window is None:
        return None
    if not is_number(window) or window < 0:
        raise MalformedEvent(
            'firmness: {} is {!r} - a staleness window is a finite non-negative number of seconds; '
            'declare a readable firmness policy, or declare none and nothing is refused on '
            'staleness'.format(PILLAR_SECONDS, window))
    return float(window)


def _hash(mapping, field, whose):
    """`mapping[field]`, asserted to be a content hash. `whose` names the side in the refusal - a
    comparison against a non-hash would answer "not firm" for the wrong reason."""
    value = mapping.get(field)
    if not is_hash(value):
        raise MalformedEvent(
            'firmness: {} {} as {!r}, which is not a content hash - the check compares the hashes '
            'a quote pinned against the ones standing now, and a missing pin is a quote that never '
            'pinned rather than a quote that went stale'.format(whose, field, value))
    return value


def _age(pillar_age):
    """The board's age in seconds, or None where it cannot be established.

    A negative age - a board stamped after the quote that read it - reads as unknown rather than
    fresh, since a future stamp would otherwise pass an arbitrarily stale board.
    """
    if pillar_age is None or not is_number(pillar_age) or pillar_age < 0:
        return None
    return float(pillar_age)
