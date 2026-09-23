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

"""Which tier a ticket falls in, and whether a verdict already stands for it.

A tiers policy is an ORDERED list and the rule is one sentence: the FIRST tier whose every declared
check passes is the one that applies. A key a tier omits is a check it does not make, so a list ends
at a catch-all declaring none, and a ticket reaching the end of a list without one falls in no tier
at all - an answer rather than a default, because a workflow nobody declared is not one a booking
may invent.

Three checks and no fourth. SIZE is a cap in the cap's OWN currency, read against the notionals the
caller states: a ticket whose notional is not stated in that currency FAILS that tier by name, which
is the safe direction and keeps every check here off the market. TENOR is the ticket's own years.
MARKET is the values vector the quote pinned, which must be the one standing under the name the tier
prices on. The two staleness checks are deliberately absent: pillar age and book staleness are
`firmness`'s two windows, enforced on every booking before any tier is read.

Pure functions over plain data - a parsed policy, a ticket and the market names standing - holding
no log, clock, store or home, so the same inputs answer the same way on the hub, on a replica and in
a gate. `assess` returns a verdict; `check` raises the same answer as `TierRefused`.
"""
from .errors import MalformedEvent, TierRefused
from .vocabulary import is_hash, is_number, is_text

#: The three checks, in the order a tier is read in. Spelled as the policy's own field names, so a
#: verdict, a refusal and a gate all name the key a desk would edit.
NOTIONAL = 'max_notional'
TENOR = 'max_tenor_years'
MARKET = 'market'
CHECKS = (NOTIONAL, TENOR, MARKET)

#: check -> what its declared bound must be for a ticket to be comparable against it. The
#: declaration parser holds a document to the same shapes; this module is handed plain data
#: everywhere else, so it asserts them itself rather than trusting the hand that built them.
BOUNDS = {NOTIONAL: lambda cap: (isinstance(cap, dict) and is_number(cap.get('amount'))
                                 and is_text(cap.get('currency'))),
          TENOR: is_number, MARKET: is_text}

#: The verdict type a standing approval must be. A rejection is the same fact the other way up.
APPROVAL = 'approval'

#: reason -> the sentence a tier fails on. A ticket over a cap, a ticket the cap cannot see and a
#: market nobody declared are three different remedies, so they are three different wordings.
FAILED = {
    'over_notional':
        'the {tier} tier caps {check} at {bound[amount]} {bound[currency]} and this ticket is '
        '{value} of it - book it under a tier that admits the size, or split the clip',
    'unstated_notional':
        'the ticket\'s notional is not stated in {bound[currency]}, the currency the {tier} tier '
        'caps {check} in - state it in that currency at the caller, since a check that crossed one '
        'currency into another would be a policy check reading a market',
    'over_tenor':
        'the {tier} tier caps {check} at {bound} years and this ticket runs {value} - book it '
        'under a tier that admits the tenor',
    'unknown_tenor':
        'the ticket carries no {check} this check could read ({value!r}), and an unknown tenor is '
        'not a tenor inside the {tier} tier\'s {bound} years',
    'undeclared_market':
        'the {tier} tier prices on the market {bound}, which nothing in this record has declared - '
        'declare the name against a values vector, or price under a tier that names no market',
    'other_market':
        'the {tier} tier prices on the market {bound}, standing on values {standing}, and this '
        'ticket pinned {value} - re-quote on the market the tier names',
}


def assess(policy, ticket, standing):
    """The verdict on one ticket: `{tier, seat, checks, refusals}`.

    `policy` is what `policy.tiers_in_force` answers. `ticket` is `{notional_in: {currency: amount},
    tenor_years, values_hash}` - the CALLER states the notional in every currency it can, so no
    check here reads a market. `standing` is `{market name: values_hash}` off the `markets` fold.

    `tier` is the first tier whose every declared check passed, and `seat` the subject an automatic
    approval signs under, `None` where the tier wants a human. `checks` reports every check read,
    per tier, with the value measured and the bound declared, so a desk shown a verdict never has to
    re-derive the comparison; `refusals` is one sentence per failure of every tier that was tried,
    which is also the route the ticket took.
    """
    for what, mapping in (('the ticket', ticket), ('the markets standing', standing),
                          ('the ticket\'s notional_in',
                           ticket.get('notional_in') if isinstance(ticket, dict) else None)):
        if not isinstance(mapping, dict):
            raise MalformedEvent(
                'tiers: {} is {}, not the object this evaluator reads - a ticket is '
                '{{notional_in, tenor_years, values_hash}} over a notional per currency, the '
                'markets standing are name -> values hash, and a check cannot be made against half '
                'of one'.format(what, type(mapping).__name__))
    for currency, amount in sorted(ticket['notional_in'].items()):
        if not is_number(amount) or amount <= 0:
            raise MalformedEvent(
                'tiers: the ticket states {!r} of {}: a notional is an AMOUNT of a currency and '
                'never a sign, so a cap cannot be compared against one - state the size the ticket '
                'deals and put which way it goes on the trade'.format(amount, currency))
    verdict = {'tier': None, 'seat': None, 'checks': [], 'refusals': []}
    for tier in _tiers(policy):
        marked = standing.get(tier.get(MARKET))
        failures = []
        for name in CHECKS:
            if name not in tier:
                continue
            value, bound, reason = _read(name, tier, ticket, standing)
            verdict['checks'].append({'tier': tier['name'], 'check': name, 'value': value,
                                      'bound': bound, 'passed': reason is None})
            if reason is not None:
                failures.append(FAILED[reason].format(
                    tier=repr(tier['name']), check=name, value=value, bound=bound,
                    standing=marked))
        if failures:
            verdict['refusals'].extend(failures)
            continue
        verdict['tier'], verdict['seat'] = tier['name'], tier.get('seat')
        return verdict
    return verdict


def check(policy, ticket, standing):
    """`assess`, raised: the verdict where a tier applies, `TierRefused` where none does.

    The refusal carries every sentence `assess` listed rather than the last one, because the tiers
    are ordered and a caller shown only the end of the route cannot see which bound to move.
    """
    verdict = assess(policy, ticket, standing)
    if verdict['tier'] is None:
        raise TierRefused('this ticket falls in no tier of the policy in force. {}'.format(
            ' AND '.join(verdict['refusals']) or 'the policy declares no tier it could fall in'))
    return verdict


def standing_approval(tier, booker, verdicts):
    """`(lsn, None)` where a verdict already approves this ticket, `(None, reason)` where none does.

    THE LATEST VERDICT STANDS, by LSN, which the record makes unique - so the order a caller hands
    them in cannot change the answer. A verdict is never withdrawn, and a rejection filed after an
    approval is what the record says last. Under `four_eyes` the approver may not be the booker,
    read off the verdict that STANDS rather than off any other row in the list. Scope is not
    re-checked here - the writer refused an unscoped approval at the append, so every verdict in the
    fold was a seat's.
    """
    if tier.get('seat') is not None:
        raise MalformedEvent(
            'tiers: the {!r} tier signs under the seat {!r}, so it asks for no standing approval - '
            'an automatic tier approves under its own seat, and demanding a human here would be a '
            'second workflow the document did not declare'.format(tier.get('name'), tier['seat']))
    if not verdicts:
        return (None, 'no verdict is filed against this ticket, and the {!r} tier names no seat of '
                      'its own - it is signed by a human or it is not signed'.format(
                          tier.get('name')))
    latest = max(verdicts, key=lambda verdict: verdict['lsn'])
    if latest['verdict'] != APPROVAL:
        return (None, 'the latest verdict on this ticket is a {} at LSN {} for {!r} - a verdict is '
                      'never withdrawn, so what stands is what was filed last; file an approval to '
                      'move it'.format(latest['verdict'], latest['lsn'], latest.get('reason')))
    if tier.get('four_eyes') and latest['actor'] == booker:
        return (None, 'the latest approval at LSN {} is {!r}\'s own and {!r} booked this ticket, '
                      'which the {!r} tier refuses: the booker and the approver are one seat - have '
                      'another seat sign it'.format(
                          latest['lsn'], latest['actor'], booker, tier.get('name')))
    return (latest['lsn'], None)


# ------------------------------------------------------------------------------------------------
# The pieces the two answers above are made of.

def _tiers(policy):
    """The ordered tiers out of `policy`, asserted to be the document a parser answered - the list
    AND every row in it, the way `firmness._windows` asserts each window rather than the pair.

    A home that declared none answers `None`, and this evaluator refuses that rather than inventing
    a workflow: what an undeclared policy means is the caller's decision. A row that is not a named
    tier refuses by name too, since a hand-built policy is what this module is handed everywhere
    outside `tiers_in_force`, and a traceback from inside a route is not a refusal.
    """
    tiers = policy.get('tiers') if isinstance(policy, dict) else None
    if not isinstance(tiers, list):
        raise MalformedEvent(
            'tiers: the policy is {}, not the ordered document `policy.tiers_in_force` answers - a '
            'home that declared no tiers has declared no workflow, and a ticket is not routed by a '
            'list this module made up'.format(type(policy).__name__))
    for position, tier in enumerate(tiers):
        if not isinstance(tier, dict) or not is_text(tier.get('name')):
            raise MalformedEvent(
                'tiers: tier {} of the policy is {!r}, not a named tier - every tier is an object '
                'declaring a name and the checks it makes, which is what the declaration parser '
                'answers; route on a document that parser read'.format(position + 1, tier))
        for check in CHECKS:
            if check in tier and not BOUNDS[check](tier[check]):
                raise MalformedEvent(
                    'tiers: the {!r} tier declares {} as {!r}, which is not a bound a ticket can be '
                    'compared against - a notional cap is {{amount, currency}}, a tenor bound is a '
                    'number and a market is a name, as the declaration parser answers them'.format(
                        tier['name'], check, tier[check]))
    return tiers


def _read(name, tier, ticket, standing):
    """One check read: `(the value measured, the bound declared, the reason it failed or None)`.

    A value the ticket does not state reads as a FAILURE of that tier rather than as a pass: a
    check that could not be made is not a check that was satisfied.
    """
    bound = tier[name]
    if name == NOTIONAL:
        value = ticket['notional_in'].get(bound['currency'])
        if not is_number(value):
            return (value, bound, 'unstated_notional')
        return (float(value), bound,
                None if float(value) <= bound['amount'] else 'over_notional')
    if name == TENOR:
        value = ticket.get('tenor_years')
        if not is_number(value):
            return (value, bound, 'unknown_tenor')
        return (float(value), bound, None if float(value) <= bound else 'over_tenor')
    pinned, marked = ticket.get('values_hash'), standing.get(bound)
    if not is_text(marked):
        return (pinned, bound, 'undeclared_market')
    return (pinned, bound, None if is_hash(pinned) and pinned == marked else 'other_market')
