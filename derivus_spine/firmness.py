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

"""Is a quote still firm, asked in two dimensions and answered separately in each.

A quote pins two hashes because it goes stale in two unrelated ways. The VALUES dimension asks
about the market: the board the price was struck on has moved, or the pin on it has aged past the
cadence that refreshes it. The PLAN dimension asks about the book: the marginal charge was solved
against a portfolio that has since moved, so the residual this trade would leave is not the one
that was priced.

The two are disjoint by measurement: a vol tick moves `values_hash` and leaves `plan_hash`
bit-identical, because quote values are the values plane while pillars, conventions and everything
a solve reads are the plan. Each dimension carries its own refusal wording, since a moved market, a
dead tick, a moved book and an aged book pin have four different remedies.

Pure functions over plain data - two pairs of hashes, two ages, two windows - holding no log, clock
or home, so the same inputs answer the same way on the hub, on a replica and in a gate. `assess`
returns a verdict; `check` raises the same answer as `QuoteNotFirm`. This does not supersede the
engine's `Quote Policy.firm_seconds`, which is the desk's own mandate and still fires as before.
"""
from .errors import MalformedEvent, QuoteNotFirm
from .vocabulary import is_hash, is_number

#: The two dimensions, named once so a refusal, a verdict and a gate all spell them the same way.
VALUES = 'values'
PLAN = 'plan'
DIMENSIONS = (VALUES, PLAN)

#: dimension -> (the hash field it compares, the window field it reads). This table plus the
#: wordings below it are the whole difference between the two dimensions.
PINNED = {VALUES: ('values_hash', 'values_seconds'), PLAN: ('plan_hash', 'plan_seconds')}

#: The refusal wording for a dimension whose pinned hash no longer matches the standing one.
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
#: The refusal wording for a pin whose age cannot be established: an unknown age is not an age
#: inside the window.
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
    and the window - so a caller can show why a quote refused without re-deriving the comparison.
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
    """`assess`, raised: the verdict when the quote is firm, `QuoteNotFirm` when it is not.

    The refusal names every dimension that failed, not the first - two stale dimensions have two
    remedies, and reporting one sends the caller back into the other.
    """
    verdict = assess(pinned, current, ages, policy)
    if not verdict['firm']:
        raise QuoteNotFirm('quote {} is no longer firm. {}'.format(
            quote_id if quote_id is not None else '(unnamed)', ' AND '.join(verdict['refusals'])))
    return verdict


def _windows(policy):
    """The two staleness windows out of `policy`, as floats. Each must be a finite non-negative
    number of seconds; anything else raises `MalformedEvent`."""
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
    """`mapping[field]`, asserted to be a content hash. `whose` names the side in the refusal - a
    comparison against a non-hash would answer "not firm" for the wrong reason."""
    value = mapping.get(field)
    if not is_hash(value):
        raise MalformedEvent(
            'firmness: {} {} as {!r}, which is not a content hash - the check compares the hashes '
            'a quote pinned against the ones standing now, and a missing pin is a quote that never '
            'pinned rather than a quote that went stale'.format(whose, field, value))
    return value


def _age(ages, dimension):
    """One dimension's age in seconds, or None where it cannot be established.

    A negative age - a clock that ran backwards between the pin and the check - reads as unknown
    rather than fresh, since a future stamp would otherwise pass an arbitrarily stale quote.
    """
    if not isinstance(ages, dict):
        raise MalformedEvent(
            'firmness: ages is {}, not the {{values, plan}} pair in seconds - each dimension has '
            'its own clock, and one number would be the conflation this check exists to '
            'refuse'.format(type(ages).__name__))
    age = ages.get(dimension)
    if age is None or not is_number(age) or age < 0:
        return None
    return float(age)
