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

import importlib
import logging
import os
from collections.abc import Mapping, Sequence

from .errors import BloombergRequestError, BloombergUnavailable, raise_response_error


def blpapi_module():
    """Return the imported `blpapi` module, raising `BloombergUnavailable` naming what to install
    when the SDK is absent. Public so a caller can refuse at startup rather than mid-request."""
    try:
        return importlib.import_module('blpapi')
    except ImportError as error:
        raise BloombergUnavailable(
            'Bloomberg blpapi is unavailable. Install the Bloomberg-supported Python SDK on '
            'this workstation and confirm the Desktop API service is running.') from error


def _error_text(element) -> str:
    return str(element).strip()


def _scalar_value(element):
    """One field element read as a scalar."""
    return element.getValue()


def _bulk_value(element):
    """One field element read as a BULK field: the list of rows it carries, each row a dict of its
    own sub-fields, or a plain `getValue()` where the field is not an array after all.

    `getValue()` on an array element returns value ZERO and raises nothing, so reading a bulk field
    with the scalar extractor truncates it to one row in silence.
    """
    if not element.isArray():
        return element.getValue()
    rows = []
    for index in range(element.numValues()):
        try:
            row = element.getValueAsElement(index)
        except Exception:
            # an array of plain scalars - the value IS the row
            rows.append(element.getValue(index))
            continue
        rows.append({str(row.getElement(position).name()): row.getElement(position).getValue()
                     for position in range(row.numElements())})
    return rows


class BloombergSession:
    """Small synchronous wrapper over Bloomberg reference data - Desktop API by default, or
    B-PIPE once an application name is given.

    B-PIPE host, port and application name default from `DV_BLOOMBERG_HOST`, `DV_BLOOMBERG_PORT`
    and `DV_BLOOMBERG_APP_NAME`, so every existing call site (`BloombergSession(timeout_ms=...)`)
    picks up a B-PIPE deployment from the environment with no code change - a Desktop API
    workstation exports none of the three and gets exactly today's `localhost:8194`, no auth.

    B-PIPE APP-ONLY AUTH IS SESSION-WIDE, not request-scoped: `AuthOptions.createWithApp` plus
    `SessionOptions.setSessionIdentityOptions` authorizes the SESSION as part of `Session.start()`,
    the same one identity then riding every request/subscription sent on it - there is no separate
    `//blp/apiauth` `AuthorizationRequest` to build by hand, the way a user-scoped identity needs.

    A B-PIPE connection or authorization failure FALLS BACK to that same Desktop API terminal
    session rather than refusing outright, so a desk with a Terminal open keeps working when
    B-PIPE itself is unreachable; the fallback is logged, never silent.
    """

    def __init__(self, host: str = None, port: int = None, timeout_ms: int = 10000,
                 connect_timeout_ms: int = None, application_name: str = None):
        self.host = host or os.environ.get('DV_BLOOMBERG_HOST', 'localhost')
        self.port = port or int(os.environ.get('DV_BLOOMBERG_PORT', 8194))
        self.timeout_ms = timeout_ms
        #: The whole budget for GETTING connected - the socket, the start attempts and the service
        #: handshake, capped together. `timeout_ms` does not reach these; it bounds `nextEvent`.
        #: None leaves the SDK's own defaults (5s x 3 attempts, then a minute of service checks).
        self.connect_timeout_ms = connect_timeout_ms
        #: `None` for a Desktop API session, which authenticates off the logged-in Terminal
        #: instead and needs no identity at all.
        self.application_name = application_name or os.environ.get('DV_BLOOMBERG_APP_NAME')
        self._api = None
        self._session = None
        self._service = None
        self._identity = None

    def start(self):
        if not self.application_name:
            return self._connect(self.host, self.port, None)
        try:
            return self._connect(self.host, self.port, self.application_name)
        except BloombergUnavailable as bpipe_error:
            logging.warning(
                'B-PIPE connection to %s:%s failed (%s) - falling back to the Desktop API '
                'terminal session at localhost:8194', self.host, self.port, bpipe_error)
            try:
                return self._connect('localhost', 8194, None)
            except BloombergUnavailable as terminal_error:
                raise BloombergUnavailable(
                    'B-PIPE at {}:{} failed ({}), and the Desktop API fallback at localhost:8194 '
                    'also failed ({})'.format(
                        self.host, self.port, bpipe_error, terminal_error)) from terminal_error

    def _connect(self, host, port, application_name):
        session = None
        try:
            api = blpapi_module()
            options = api.SessionOptions()
            options.setServerHost(host)
            options.setServerPort(port)
            if self.connect_timeout_ms is not None:
                # one attempt, not the SDK's three - the per-attempt timeout is otherwise
                # multiplied by the retries and the backoff between them
                options.setConnectTimeout(self.connect_timeout_ms)
                options.setNumStartAttempts(1)
                options.setServiceCheckTimeout(self.connect_timeout_ms)
                options.setServiceDownloadTimeout(self.connect_timeout_ms)
            correlation_id = None
            if application_name:
                correlation_id = api.CorrelationId(application_name)
                options.setSessionIdentityOptions(
                    api.AuthOptions.createWithApp(application_name), correlation_id)
            session = api.Session(options)
            if not session.start():
                raise BloombergUnavailable(
                    'Bloomberg session did not start at {}:{}'.format(host, port))
            if not session.openService('//blp/refdata'):
                raise BloombergUnavailable('Bloomberg service //blp/refdata could not be opened')
            identity = self._await_authorization(session, api) if application_name else None
            self._api = api
            self._session = session
            self._service = session.getService('//blp/refdata')
            self._identity = identity
            return self
        except BloombergUnavailable:
            if session is not None:
                session.stop()
            raise
        except Exception as error:
            if session is not None:
                session.stop()
            raise BloombergUnavailable('Bloomberg session failed: {}'.format(error)) from error

    def _await_authorization(self, session, api):
        """The session-wide identity `setSessionIdentityOptions` requested at construction,
        confirmed by its own `AUTHORIZATION_STATUS` event rather than a request built here - the
        app-only identity authorizes the SESSION, so `session.getAuthorizedIdentity()` is what
        every later `sendRequest` on it carries."""
        while True:
            event = session.nextEvent(self.timeout_ms)
            if event.eventType() == api.Event.TIMEOUT:
                raise BloombergUnavailable('Bloomberg authorization request timed out')
            for message in event:
                if message.messageType() == api.Name('AuthorizationSuccess'):
                    return session.getAuthorizedIdentity()
                if message.messageType() in (
                        api.Name('AuthorizationFailure'), api.Name('AuthorizationRevoked')):
                    raise BloombergUnavailable(
                        'Bloomberg authorization failed: {}'.format(message))

    def stop(self) -> None:
        if self._session is not None:
            self._session.stop()
        self._api = self._session = self._service = None
        self._identity = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def reference_data(self, securities: Sequence[str], fields: Sequence[str],
                       overrides: Mapping[str, object] = None) -> dict[str, dict[str, object]]:
        """`{security: {field: value}}`, refusing the whole batch on ANY per-security error: a
        production tick built from a partial answer is a wrong market, not a smaller one."""
        response = {}
        for security, error, values in self._walked(securities, fields, overrides):
            if error is not None:
                raise_response_error('{}: {}'.format(security, error))
            response[security] = values
        return response

    def reference_data_report(self, securities: Sequence[str], fields: Sequence[str],
                              overrides: Mapping[str, object] = None
                              ) -> dict[str, dict[str, object]]:
        """Per-security outcomes, for DISCOVERY: `{security: {'ok', 'error', 'fields'}}` with every
        requested name answered, a refused ticker reported rather than raised. A request-level
        error - a timeout, a `responseError` - still raises: that is transport, not a name."""
        return self._reported(securities, self._walked(securities, fields, overrides))

    def bulk_reference_data_report(self, securities: Sequence[str], fields: Sequence[str],
                                   overrides: Mapping[str, object] = None
                                   ) -> dict[str, dict[str, object]]:
        """`reference_data_report`'s contract over BULK fields: same `{security: {'ok', 'error',
        'fields'}}`, same tolerance, but each field answers a LIST OF ROWS rather than one value.

        Separate from the scalar reader because a field means something different here, not because
        the policy differs; both walk one `_request` under their own extractor.
        """
        return self._reported(securities, self._walked_bulk(securities, fields, overrides))

    @staticmethod
    def _reported(securities, walked):
        report = {}
        for security, error, values in walked:
            report[security] = {'ok': error is None, 'error': error, 'fields': values}
        for security in securities:
            report.setdefault(security, {'ok': False, 'error': 'no answer in the response',
                                         'fields': {}})
        return report

    def _walked(self, securities, fields, overrides=None):
        """The event walk the SCALAR readers share: `(security, error, values)` per name, `error`
        carrying Bloomberg's own text where it refused one.

        Materialized, so the response is DRAINED before either policy raises and the session is
        left clean for its next request. The cost is that a transport failure on a later event
        outranks a per-security error already walked."""
        return self._drained(self._walking(self._walk, securities, fields, overrides))

    def _walked_bulk(self, securities, fields, overrides=None):
        """`_walked` over the bulk extractor - the same drain, wrapping and failure precedence."""
        return self._drained(self._walking(self._walk_bulk, securities, fields, overrides))

    @staticmethod
    def _walking(walk, securities, fields, overrides):
        """The walk, bound to its request. `overrides` is passed ONLY when there are any, so a
        subclass overriding the walk with the two arguments it has always taken - the canned walks
        every offline gate in this package is built on - keeps working."""
        if overrides:
            return lambda: walk(securities, fields, overrides)
        return lambda: walk(securities, fields)

    def _drained(self, walk):
        if self._session is None or self._service is None or self._api is None:
            raise BloombergUnavailable('BloombergSession must be started before requesting data')
        try:
            return list(walk())
        except (BloombergRequestError, BloombergUnavailable):
            raise
        except Exception as error:
            raise BloombergRequestError('Bloomberg reference-data request failed: {}'.format(error)) from error

    def _walk(self, securities, fields, overrides=None):
        yield from self._request(securities, fields, _scalar_value, overrides)

    def _walk_bulk(self, securities, fields, overrides=None):
        yield from self._request(securities, fields, _bulk_value, overrides)

    def _request(self, securities, fields, value_of, overrides=None):
        """`overrides` is `{fieldId: value}` - the request-level parameters a field reads in place
        of its default (`IVOL_MATURITY`, `BEST_FPERIOD_OVERRIDE` and their kind), sent as
        Bloomberg's own `overrides` array. Without one, a field whose NAME carries a tenor still
        answers at the service's default maturity."""
        request = self._service.createRequest('ReferenceDataRequest')
        security_element = request.getElement('securities')
        field_element = request.getElement('fields')
        for security in securities:
            security_element.appendValue(security)
        for field in fields:
            field_element.appendValue(field)
        if overrides:
            override_element = request.getElement('overrides')
            for field, value in overrides.items():
                row = override_element.appendElement()
                row.setElement('fieldId', str(field))
                row.setElement('value', str(value))
        # `identity` rides ONLY under B-PIPE auth, so every canned two-argument `sendRequest` this
        # package's own gates are built on - never authenticated - keeps working unchanged.
        if self._identity is not None:
            self._session.sendRequest(request, identity=self._identity)
        else:
            self._session.sendRequest(request)

        while True:
            event = self._session.nextEvent(self.timeout_ms)
            if event.eventType() == self._api.Event.TIMEOUT:
                raise BloombergRequestError('Bloomberg reference-data request timed out')
            for message in event:
                if message.hasElement('responseError'):
                    raise_response_error(_error_text(message.getElement('responseError')))
                if not message.hasElement('securityData'):
                    continue
                security_data = message.getElement('securityData')
                for index in range(security_data.numValues()):
                    item = security_data.getValueAsElement(index)
                    security = item.getElementAsString('security')
                    error = None
                    if item.hasElement('securityError'):
                        error = _error_text(item.getElement('securityError'))
                    elif item.hasElement('fieldExceptions'):
                        exceptions = item.getElement('fieldExceptions')
                        if exceptions.numValues():
                            error = _error_text(exceptions)
                    values = {}
                    if item.hasElement('fieldData'):
                        data = item.getElement('fieldData')
                        values = {field: value_of(data.getElement(field))
                                  for field in fields if data.hasElement(field)}
                    yield security, error, values
            if event.eventType() == self._api.Event.RESPONSE:
                return