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

"""`BloombergSession`'s own connection, B-PIPE auth and mktdata snapshot mechanics, against a FAKE
blpapi - no real SDK or network reachable from a test box. The service's gates price off the same
fake as a canned terminal (`canned_terminal`), its reference data declared as data.

The fake scripts `Session.start()`, `openService()` and the `AUTHORIZATION_STATUS` event
`SessionOptions.setSessionIdentityOptions` schedules, keyed by `(host, port)` - a B-PIPE attempt
and its Desktop API fallback are two DIFFERENT sessions in one script, exactly as
`BloombergSession.start` builds them one after the other. `images` scripts what a subscription to
each topic answers: a dict of mktdata fields, `('fail', category, description)`, or nothing at all.
`refdata` scripts what a `ReferenceDataRequest` answers for each security: a dict of fields, the
text of the `securityError` it is refused with, or nothing at all.

The fake enters as the `blpapi` module the package imports - an entry in `sys.modules`, which is
where `blpapi_module()` finds the SDK - so nothing in the package is rebound.
"""
import datetime
import sys
import types

import pytest

from derivus_bloomberg import session
from derivus_bloomberg.errors import BloombergRequestError, BloombergUnavailable

APP = 'Investec_SA:TSSPricingTool'
BPIPE = ('10.0.88.31', 8194)
DESKTOP = ('localhost', 8194)
#: What a session reads from the environment, which is the test's to say and not the shell's.
ENVIRONMENT = ('DV_BLOOMBERG_HOST', 'DV_BLOOMBERG_PORT', 'DV_BLOOMBERG_APP_NAME')


@pytest.fixture(autouse=True)
def no_bloomberg_environment(monkeypatch):
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


class FakeName:
    """`blpapi.Name` - a value, compared by it, so a scripted message and a call to `api.Name(...)`
    made afterwards still agree."""
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, FakeName) and other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return 'Name({!r})'.format(self.value)


class FakeMessage:
    def __init__(self, message_type):
        self._type = message_type

    def messageType(self):
        return self._type

    def __str__(self):
        return 'FakeMessage({!r})'.format(self._type)


class FakeCorrelation:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeElement:
    def __init__(self, value):
        self._value = value

    def numValues(self):
        return 0 if self._value is None else 1

    def getValue(self):
        return self._value

    def __str__(self):
        return str(self._value)

    getValueAsString = __str__


class FakeData:
    """A response element: a field by name, a row by index, a scalar as a `FakeElement`."""
    def __init__(self, value):
        self._value = value

    def hasElement(self, name):
        return name in self._value

    def getElement(self, name):
        value = self._value[name]
        return FakeData(value) if isinstance(value, (dict, list)) else FakeElement(value)

    def getElementAsString(self, name):
        return str(self._value[name])

    def numValues(self):
        return len(self._value)

    def getValueAsElement(self, index):
        return FakeData(self._value[index])


class FakeDataMessage(FakeMessage):
    def __init__(self, topic, fields):
        super().__init__(FakeName('MarketDataEvents'))
        self._topic, self._fields = topic, fields

    def correlationIds(self):
        return [FakeCorrelation(self._topic)]

    def asElement(self):
        return FakeData(self._fields)


class FakeFailureMessage(FakeMessage):
    def __init__(self, topic, category, description):
        super().__init__(FakeName('SubscriptionFailure'))
        self._topic = topic
        self._text = ('SubscriptionFailure = { reason = { category = "%s" errorCode = 7 '
                      'description = "%s" } }' % (category, description))

    def correlationIds(self):
        return [FakeCorrelation(self._topic)]

    def __str__(self):
        return self._text


class FakeSubscriptionList:
    def __init__(self):
        self.topics, self.fields = [], []

    def add(self, topic, fields, options, correlation_id):
        self.topics.append(topic)
        self.fields.append(list(fields))


class FakeEvent:
    TIMEOUT = 'TIMEOUT'
    AUTHORIZATION_STATUS = 'AUTHORIZATION_STATUS'
    RESPONSE = 'RESPONSE'
    SUBSCRIPTION_DATA = 'SUBSCRIPTION_DATA'
    SUBSCRIPTION_STATUS = 'SUBSCRIPTION_STATUS'

    def __init__(self, event_type, messages=()):
        self._type = event_type
        self._messages = list(messages)

    def eventType(self):
        return self._type

    def __iter__(self):
        return iter(self._messages)


class FakeSessionOptions:
    def __init__(self):
        self.host = None
        self.port = None
        self.identity_options = None
        self.correlation_id = None

    def setServerHost(self, host):
        self.host = host

    def setServerPort(self, port):
        self.port = port

    def setConnectTimeout(self, ms):
        pass

    def setNumStartAttempts(self, n):
        pass

    def setServiceCheckTimeout(self, ms):
        pass

    def setServiceDownloadTimeout(self, ms):
        pass

    def setSessionIdentityOptions(self, auth_options, correlation_id):
        self.identity_options = auth_options
        self.correlation_id = correlation_id


class FakeRequest:
    """A `ReferenceDataRequest`: the securities and the fields appended to it."""
    def __init__(self):
        self.elements = {'securities': [], 'fields': []}

    def getElement(self, name):
        return types.SimpleNamespace(appendValue=self.elements[name].append)


class FakeSession:
    def __init__(self, options, script, asked):
        self.options = options
        self.stopped = False
        config = script.get((options.host, options.port), {})
        self._start_ok = config.get('start', True)
        self._open_service_ok = config.get('open_service', True)
        self._events = list(config.get('events', []))
        self._images = config.get('images', {})
        self._refdata = config.get('refdata', {})
        self._answering = config.get('answering')
        self.asked = asked
        self.subscriptions = []
        self.unsubscribed = 0
        self._identity = ('Identity', options.host, options.port)

    def start(self):
        return self._start_ok

    def openService(self, name):
        return self._open_service_ok

    def getService(self, name):
        """The session serves its requests itself."""
        return self

    def createRequest(self, kind):
        return FakeRequest()

    def nextEvent(self, timeout_ms):
        if not self._events:
            return FakeEvent(FakeEvent.TIMEOUT)
        return self._events.pop(0)

    def subscribe(self, subscriptions):
        self.subscriptions.append(subscriptions)
        for topic in subscriptions.topics:
            image = self._images.get(topic)
            if image is None:
                continue
            if isinstance(image, tuple):
                self._events.append(FakeEvent(FakeEvent.SUBSCRIPTION_STATUS, [
                    FakeFailureMessage(topic, image[1], image[2])]))
            else:
                self._events.append(FakeEvent(FakeEvent.SUBSCRIPTION_DATA, [
                    FakeDataMessage(topic, image)]))

    def unsubscribe(self, subscriptions):
        self.unsubscribed += 1

    def getAuthorizedIdentity(self):
        return self._identity

    def sendRequest(self, request, identity=None):
        """One `RESPONSE`, a row per security `refdata` declares; `answering()` runs first."""
        securities = request.elements['securities']
        self.asked.extend(securities)
        if self._answering is not None:
            self._answering()
        rows = [{'security': name,
                 'securityError' if isinstance(answer, str) else 'fieldData': answer}
                for name, answer in ((name, self._refdata.get(name)) for name in securities)
                if answer is not None]
        self._events.append(FakeEvent(FakeEvent.RESPONSE, [FakeData({'securityData': rows})]))

    def stop(self):
        self.stopped = True


def make_fake(script):
    """A fake `blpapi` module, scripted by `{(host, port): {'start', 'open_service', 'events',
    'images', 'refdata', 'answering'}}` - everything a `BloombergSession` reads, and nothing else.
    `asked` is every security a reference-data request named, in order."""

    class _AuthOptions:
        @staticmethod
        def createWithApp(name):
            return ('AuthOptionsApp', name)

    class Fake:
        Event = FakeEvent
        Name = staticmethod(FakeName)
        AuthOptions = _AuthOptions

        def __init__(self):
            self.sessions, self.asked = [], []

        def SessionOptions(self):
            return FakeSessionOptions()

        def Session(self, options):
            fake_session = FakeSession(options, script, self.asked)
            self.sessions.append(fake_session)
            return fake_session

        def CorrelationId(self, value):
            return FakeCorrelation(value)

        def SubscriptionList(self):
            return FakeSubscriptionList()

    return Fake()


def canned_terminal(monkeypatch, refdata, answering=None):
    """A Desktop API terminal answering `refdata`, entered as the `blpapi` module the package
    imports, and the session's endpoint left to its defaults; `answering()`, where given, runs as
    each request lands. The fake is returned, `refdata` on it as declared."""
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    fake = make_fake({DESKTOP: {'refdata': refdata, 'answering': answering}})
    fake.refdata = refdata
    monkeypatch.setitem(sys.modules, 'blpapi', fake)
    return fake


def test_desktop_api_session_asks_for_no_identity_and_never_waits_on_auth(monkeypatch):
    """A Terminal session (`application_name=None`) authenticates off the logged-in Desktop and
    must never touch `setSessionIdentityOptions` or the `nextEvent` auth wait - a mutant that
    always waits for `AUTHORIZATION_STATUS` would hang here with no scripted event to answer it.

    Killing mutation: the identity options built on every session, the Terminal's included."""
    fake = make_fake({DESKTOP: {'start': True, 'open_service': True}})
    monkeypatch.setitem(sys.modules, 'blpapi', fake)

    bloomberg = session.BloombergSession(host='localhost', port=8194).start()

    assert bloomberg._identity is None
    assert fake.sessions[0].options.identity_options is None
    bloomberg.stop()


def test_leaving_the_bpipe_env_vars_unset_is_the_terminal_and_setting_them_is_bpipe(monkeypatch):
    """The only switch between the two deployments is whether the env vars are declared.

    Killing mutation: `DV_BLOOMBERG_APP_NAME` unread - the declared deployment runs as a Terminal."""
    terminal = session.BloombergSession()
    assert (terminal.host, terminal.port, terminal.application_name) == ('localhost', 8194, None)

    monkeypatch.setenv('DV_BLOOMBERG_HOST', BPIPE[0])
    monkeypatch.setenv('DV_BLOOMBERG_PORT', str(BPIPE[1]))
    monkeypatch.setenv('DV_BLOOMBERG_APP_NAME', APP)
    bpipe = session.BloombergSession()
    assert (bpipe.host, bpipe.port, bpipe.application_name) == (BPIPE[0], BPIPE[1], APP)


def test_bpipe_session_authorizes_via_session_wide_identity_options(monkeypatch):
    """B-PIPE app-only auth is SESSION-wide: `AuthOptions.createWithApp` rides on
    `setSessionIdentityOptions` at construction, and success is confirmed by its OWN
    `AUTHORIZATION_STATUS` event rather than a hand-built `//blp/apiauth` request. Killing a
    mutant that reverts to the request-scoped `sendAuthorizationRequest` flow, which this fake
    exposes no service for at all.

    Killing mutation: the identity options built off a fixed name rather than the application's."""
    fake = make_fake({BPIPE: {'start': True, 'open_service': True, 'events': [
        FakeEvent(FakeEvent.AUTHORIZATION_STATUS, [FakeMessage(FakeName('AuthorizationSuccess'))])
    ]}})
    monkeypatch.setitem(sys.modules, 'blpapi', fake)

    bloomberg = session.BloombergSession(
        host=BPIPE[0], port=BPIPE[1], application_name=APP).start()

    connected = fake.sessions[0]
    assert (connected.options.host, connected.options.port) == BPIPE
    assert connected.options.identity_options == ('AuthOptionsApp', APP), (
        'the identity options must be built off THIS application name, not a generic one')
    assert bloomberg._identity == connected.getAuthorizedIdentity(), (
        'the session-wide identity, not a request-scoped one nothing here built')
    bloomberg.stop()


@pytest.mark.parametrize('failure', ['AuthorizationFailure', 'AuthorizationRevoked'])
def test_bpipe_authorization_failure_falls_back_to_desktop_api(monkeypatch, failure):
    """Either way B-PIPE can refuse the session identity, the fallback is the SAME plain Desktop
    API session `start()` always tries with no application name - the failed B-PIPE session is
    stopped, never reused, and the fallback carries no B-PIPE identity of its own.

    Killing mutation: a session refused on authorization left running."""
    fake = make_fake({
        BPIPE: {'start': True, 'open_service': True,
               'events': [FakeEvent(FakeEvent.AUTHORIZATION_STATUS, [FakeMessage(FakeName(failure))])]},
        DESKTOP: {'start': True, 'open_service': True},
    })
    monkeypatch.setitem(sys.modules, 'blpapi', fake)

    bloomberg = session.BloombergSession(
        host=BPIPE[0], port=BPIPE[1], application_name=APP).start()

    assert bloomberg._identity is None, 'the Desktop API fallback carries no B-PIPE identity'
    assert len(fake.sessions) == 2, 'the failed B-PIPE session and its Desktop API fallback'
    failed, fallback = fake.sessions
    assert failed.stopped, 'a session refused on authorization must not be left running'
    assert (fallback.options.host, fallback.options.port) == DESKTOP
    assert fallback.options.identity_options is None, 'the fallback asks for no B-PIPE identity'
    bloomberg.stop()


def test_bpipe_and_desktop_api_both_failing_names_both_in_one_refusal(monkeypatch):
    """Neither leg reachable is not a B-PIPE-shaped error alone: the refusal names both attempts,
    so an operator is not left guessing which of the two is actually down.

    Killing mutation: the Terminal's own refusal raised alone."""
    fake = make_fake({BPIPE: {'start': False}, DESKTOP: {'start': False}})
    monkeypatch.setitem(sys.modules, 'blpapi', fake)

    with pytest.raises(BloombergUnavailable, match='Desktop API fallback'):
        session.BloombergSession(host=BPIPE[0], port=BPIPE[1], application_name=APP).start()


PRICES = ['PX_LAST', 'PX_BID', 'PX_ASK', 'LAST_UPDATE_DT']
AUTHORIZED = FakeEvent(FakeEvent.AUTHORIZATION_STATUS, [FakeMessage(FakeName('AuthorizationSuccess'))])


def today():
    """The UTC date a live quote's time of day is read as."""
    return datetime.datetime.now(datetime.timezone.utc).date()


def bpipe(monkeypatch, images):
    """A started B-PIPE session over the fake, scripted with `images`, and the fake itself. The
    only session the script holds is B-PIPE's: a connect to anywhere else finds nothing."""
    fake = make_fake({BPIPE: {'events': [AUTHORIZED], 'images': images}})
    monkeypatch.setitem(sys.modules, 'blpapi', fake)
    return session.BloombergSession(host=BPIPE[0], port=BPIPE[1], application_name=APP).start(), fake


def test_bpipe_prices_are_read_off_a_mktdata_snapshot_under_their_reference_names(monkeypatch):
    """B-PIPE is entitled to `//blp/mktdata` and not `//blp/refdata`, so `PX_LAST`/`PX_BID`/`PX_ASK`
    come off the first image of a subscription as `LAST_PRICE`/`BID`/`ASK`. Distinct values per
    field kill a mapping that crosses them; the mid-only fixing carries no `PX_BID` rather than an
    error; a live quote's `LAST_UPDATE_DT` is today and a daily fixing's is the date it printed.

    Killing mutation: `PX_BID` and `PX_ASK` mapped onto each other's mktdata field."""
    bloomberg, fake = bpipe(monkeypatch, {
        'USDZAR BGN Curncy': {'LAST_PRICE': 16.5758, 'BID': 16.573, 'ASK': 16.5786, 'TIME': '13:30:42'},
        'SOFRRATE Index': {'LAST_PRICE': 3.9, 'TIME': '2026-09-30'}})

    report = bloomberg.reference_data_report(['USDZAR BGN Curncy', 'SOFRRATE Index'], PRICES)

    assert report['USDZAR BGN Curncy'] == {'ok': True, 'error': None, 'fields': {
        'PX_LAST': 16.5758, 'PX_BID': 16.573, 'PX_ASK': 16.5786, 'LAST_UPDATE_DT': today()}}
    assert report['SOFRRATE Index'] == {'ok': True, 'error': None, 'fields': {
        'PX_LAST': 3.9, 'LAST_UPDATE_DT': datetime.date(2026, 9, 30)}}
    subscription = fake.sessions[0].subscriptions[0]
    assert subscription.fields[0] == ['LAST_PRICE', 'BID', 'ASK', 'TIME']
    assert fake.sessions[0].unsubscribed == 1, 'an image is read once and the stream let go'


def test_a_staleness_check_alone_subscribes_to_the_time_and_nothing_else(monkeypatch):
    """Killing mutation: every snapshot subscribing the three prices beside the time."""
    bloomberg, fake = bpipe(monkeypatch, {'USDZARV3M BGN Curncy': {'TIME': '13:25:39'}})

    report = bloomberg.reference_data_report(['USDZARV3M BGN Curncy'], ['LAST_UPDATE_DT'])

    assert report['USDZARV3M BGN Curncy']['fields'] == {'LAST_UPDATE_DT': today()}
    assert fake.sessions[0].subscriptions[0].fields[0] == ['TIME']


def test_what_bpipe_cannot_price_is_reported_by_name_and_the_rest_stand(monkeypatch):
    """ZARONIA is refused outright (NOT_ENTITLED), JIBA3M never answers and the swaption cell has
    no price in its image. A B-PIPE session does not reach for a terminal: each is reported with
    Bloomberg's own reason, the quote that WAS answered stands for the tolerant reader, and the
    strict one a production tick uses refuses the whole batch, a partial market being a wrong one.
    Only the session B-PIPE opened exists - a connect to `localhost` would find nothing.

    Killing mutation: a `SubscriptionFailure` not read - the refusal reads as a missing image."""
    bloomberg, fake = bpipe(monkeypatch, {
        'USDZAR BGN Curncy': {'LAST_PRICE': 16.5, 'TIME': '13:30:42'},
        'ZARONIA Index': ('fail', 'NOT_ENTITLED', 'Security Entitlement Check Failed! EID(s) needed: 17567'),
        'SASN0A1 Curncy': {'TIME': '13:30:42'}})
    asked = ['USDZAR BGN Curncy', 'ZARONIA Index', 'JIBA3M Index', 'SASN0A1 Curncy']

    report = bloomberg.reference_data_report(asked, ['PX_LAST'])

    assert report['USDZAR BGN Curncy'] == {'ok': True, 'error': None, 'fields': {'PX_LAST': 16.5}}
    assert 'EID(s) needed: 17567' in report['ZARONIA Index']['error']
    assert 'no mktdata image' in report['JIBA3M Index']['error']
    assert 'carries no price' in report['SASN0A1 Curncy']['error']
    assert not any(report[name]['ok'] for name in asked[1:])
    with pytest.raises(BloombergRequestError, match='ZARONIA Index'):
        bloomberg.reference_data(asked, ['PX_LAST'])
    assert len(fake.sessions) == 1, 'a B-PIPE session opened a second one'


@pytest.mark.parametrize('call,fields,overrides,named', [
    ('reference_data_report', ['NAME', 'PX_LAST'], None, 'NAME'),
    ('reference_data_report', ['PX_LAST'], {'IVOL_MATURITY': '3M'}, 'an override'),
    ('bulk_reference_data_report', ['CHAIN_TICKERS'], None, 'CHAIN_TICKERS')])
def test_a_request_mktdata_cannot_express_refuses_by_name_on_bpipe(
        monkeypatch, call, fields, overrides, named):
    """A field outside the four prices, an override and a bulk field have no mktdata form, so a
    B-PIPE session refuses them by name rather than quietly asking a terminal - no subscription is
    made for any of it.

    Killing mutations: the scalar guard dropped - a subscription is made and nothing refuses; the
    bulk guard dropped - the request goes to a refdata service B-PIPE does not carry."""
    bloomberg, fake = bpipe(monkeypatch, {})

    with pytest.raises(BloombergRequestError, match=named):
        getattr(bloomberg, call)(['SPX Index'], fields, *([overrides] if overrides else []))

    assert fake.sessions[0].subscriptions == []


def test_a_session_with_no_bpipe_runs_wholly_on_the_terminal(monkeypatch):
    """B-PIPE is checked once, at start. Where it is not running the session is a Desktop API one
    for good - no identity, no mktdata snapshot, every request the plain reference-data one - so a
    user on the go with only a terminal is exactly as served as before B-PIPE existed.

    Killing mutation: the fallback connecting with the application's identity."""
    fake = make_fake({BPIPE: {'start': False}, DESKTOP: {'start': True, 'open_service': True}})
    monkeypatch.setitem(sys.modules, 'blpapi', fake)

    bloomberg = session.BloombergSession(
        host=BPIPE[0], port=BPIPE[1], application_name=APP).start()

    assert bloomberg._identity is None
    assert [(s.options.host, s.options.port) for s in fake.sessions] == [BPIPE, DESKTOP]
    assert fake.sessions[1].subscriptions == []
