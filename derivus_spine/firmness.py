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

"""Is this quote still firm - asked in TWO dimensions, and answered separately in each.

A quote pins two hashes because it can go stale in two unrelated ways, and an approval that checked
one of them would be an approval that let the other through. The VALUES dimension asks about the
market: the board the price was struck on has moved, or the pin on it has aged past the cadence
that refreshes it. The PLAN dimension asks about the book: the marginal charge was solved against a
portfolio, and that portfolio has moved since - somebody booked, somebody amended, the base date
rolled - so the residual this trade would actually leave is not the one that was priced.

They are DISJOINT by measurement rather than by assumption, and that is an engine property this
module is the consumer of. Since the `Market Prices` partition landed, a vol tick moves
`values_hash` and leaves `plan_hash` bit-identical: quote values are the values plane, pillars and
conventions and everything a solve reads is the plan. So a market that ticked under a standing
quote trips the values dimension and CANNOT trip the plan one, and the gate says so on a fixture
that ticks a vol with no booking in sight. Conflating the two would produce exactly the failure the
brief names - an aged quote approvable at a dead market's solve - dressed as a single "staleness"
number nobody could act on.

Four refusals, two per dimension, each naming its own remedy, because a desk does different things
about them. A moved market means re-quote at the market that is standing. An aged pin means the
cadence stopped and somebody should look at the terminal before re-quoting. A moved book means the
charge has to be re-solved against the book this trade would now join. An aged book pin is the
backstop under all of it.

This is a PURE FUNCTION over plain data - two pairs of hashes, two ages, two windows - and it holds
no log, no clock and no home. Everything it needs was folded out of the record before it was called
(`policy.firmness_in_force`) and the caller measured the ages against its own clock, so the same
inputs answer the same way on the hub, on a replica and in a gate. `assess` answers a verdict;
`check` is the same answer raised as the refusal a booking meets. The engine's existing
`Quote Policy.firm_seconds` clock is NOT superseded by any of this: it is the desk's own mandate
about how long its salespeople may stand behind a price, it keeps firing exactly as it did, and it
is named beside these two dimensions rather than replaced by them - a desk window and a provenance
window are two different promises to two different people.
"""
from .errors import MalformedEvent, QuoteNotFirm
from .vocabulary import is_hash, is_number

#: The two dimensions, named once so a refusal, a verdict and a gate all spell them the same way.
VALUES = 'values'
PLAN = 'plan'
DIMENSIONS = (VALUES, PLAN)

#: dimension -> (the hash field it compares, the window field it reads). The whole difference
#: between the two dimensions is this table plus the sentences below it.
PINNED = {VALUES: ('values_hash', 'values_seconds'), PLAN: ('plan_hash', 'plan_seconds')}

#: What each dimension means when it refuses - the sentence a salesperson reads. Kept here rather
#: than inline so the two remedies stay visibly different from each other.
MOVED = {
    VALUES: 'the market moved under it: the quote was struck on values {pinned} and the board '
            'standing now is {current} - re-quote, because the price that was given belongs to a '
            'market that is no longer the one this booking would land against',
    PLAN: 'the book moved under it: the charge was solved against plan {pinned} and the book '
          'standing now is {current} - re-quote, because the residual this trade leaves is a '
          'property of the portfolio it joins, and that portfolio is not the one that was priced',
}
AGED = {
    VALUES: 'the market pin is {age:.1f}s old against a {window:.0f}s window - re-quote, and check '
            'the tick: a values pin older than the cadence that refreshes it was taken off a board '
            'nobody is watching',
    PLAN: 'the book pin is {age:.1f}s old against a {window:.0f}s window - re-quote: the book moves '
          'by booking rather than by clock, so a pin this old is one nobody has compared against '
          'the book since',
}
#: An age nobody can establish, which is not an age inside a window - the edge's own ruling about a
#: pending file with no stamp, met here in the general case.
UNKNOWN = ('the {dimension} pin carries no age this check could establish, and an unknown age is '
           'not an age inside the {window:.0f}s window - re-quote: the pin has had an unknown '
           'number of seconds to go stale')


def assess(pinned, current, ages, policy):
    """The verdict on one quote: `{firm, refusals, values: {...}, plan: {...}}`.

    `pinned` and `current` are `{'plan_hash', 'values_hash'}` - what the quote pinned, and what the
    book and its market are right now. `ages` is `{'values', 'plan'}` in seconds, or None in a
    dimension whose age cannot be established. `policy` is `{'values_seconds', 'plan_seconds'}`, as
    `policy.firmness_in_force` answers it.

    Each dimension is answered on its own and reports everything it read - the two hashes, the age
    and the window - so a caller can show a desk why a quote refused without re-deriving the
    comparison, and so a verdict is a document rather than a boolean somebody has to trust.
    """
    for name, mapping in (('pinned', pinned), ('current', current)):
        if not isinstance(mapping, dict):
            raise MalformedEvent(
                'firmness: {} is {}, not the {{plan_hash, values_hash}} pair a quote pins - the '
                'check compares two pairs and cannot be asked about half of one'.format(
                    name, type(mapping).__name__))
    windows = _windows(policy)
    verdict = {'firm': True, 'refusals': []}
    for dimension in DIMENSIONS:
        field, window_field = PINNED[dimension]
        window = windows[window_field]
        was, now = _hash(pinned, field, 'the quote pinned'), _hash(current, field, 'the book shows')
        age = _age(ages, dimension)
        refusals = []
        if was != now:
            refusals.append(MOVED[dimension].format(pinned=was, current=now))
        if age is None:
            refusals.append(UNKNOWN.format(dimension=dimension, window=window))
        elif age > window:
            refusals.append(AGED[dimension].format(age=age, window=window))
        verdict[dimension] = {'firm': not refusals, 'pinned': was, 'current': now,
                              'age': age, 'window': window,
                              'refusals': ['the {} dimension: {}'.format(dimension, said)
                                           for said in refusals]}
        verdict['refusals'].extend(verdict[dimension]['refusals'])
        verdict['firm'] = verdict['firm'] and verdict[dimension]['firm']
    return verdict


def check(pinned, current, ages, policy, quote_id=None):
    """`assess`, raised. Answers the verdict when the quote is firm; `QuoteNotFirm` naming EVERY
    dimension that refused when it is not.

    Every one of them, not the first: a quote that is stale on both dimensions has two things wrong
    with it and two remedies, and reporting one of them sends a salesperson back to re-quote into
    the same refusal.
    """
    verdict = assess(pinned, current, ages, policy)
    if not verdict['firm']:
        raise QuoteNotFirm('quote {} is no longer firm. {}'.format(
            quote_id if quote_id is not None else '(unnamed)', ' AND '.join(verdict['refusals'])))
    return verdict


def _windows(policy):
    """The two windows, checked. A policy that will not read is refused here rather than compared
    against, because a window nobody can read is one no approval could be measured against."""
    if not isinstance(policy, dict):
        raise MalformedEvent(
            'firmness: the policy is {}, not the {{values_seconds, plan_seconds}} pair - hand in '
            'what `policy.firmness_in_force` answers, which is the declared document or the stated '
            'defaults'.format(type(policy).__name__))
    windows = {}
    for _, window_field in PINNED.values():
        window = policy.get(window_field)
        if not is_number(window) or window < 0:
            raise MalformedEvent(
                'firmness: {} is {!r} - a staleness window is a finite non-negative number of '
                'seconds; declare a readable {} policy or pass the stated defaults'.format(
                    window_field, window, 'firmness'))
        windows[window_field] = float(window)
    return windows


def _hash(mapping, field, whose):
    """One pinned or standing hash, or a refusal naming which side was not one. A comparison
    between a hash and something that is not one would answer "not firm" for the wrong reason."""
    value = mapping.get(field)
    if not is_hash(value):
        raise MalformedEvent(
            'firmness: {} {} as {!r}, which is not a content hash - the check compares the hashes '
            'a quote pinned against the ones standing now, and a missing pin is a quote that never '
            'pinned rather than a quote that went stale'.format(whose, field, value))
    return value


def _age(ages, dimension):
    """One dimension's age in seconds, or None where it cannot be established. A negative age is a
    clock that ran backwards between the pin and the check, and it reads as unknown rather than as
    fresh - a future stamp is the one reading that would let an arbitrarily stale quote through."""
    if not isinstance(ages, dict):
        raise MalformedEvent(
            'firmness: ages is {}, not the {{values, plan}} pair in seconds - each dimension has '
            'its own clock, and one number would be the conflation this check exists to '
            'refuse'.format(type(ages).__name__))
    age = ages.get(dimension)
    if age is None or not is_number(age) or age < 0:
        return None
    return float(age)
