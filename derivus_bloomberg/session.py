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
from collections.abc import Sequence

from .errors import BloombergRequestError, BloombergUnavailable, raise_response_error


def blpapi_module():
    """The Bloomberg SDK, or the refusal that names what is missing. Public because a CALLER may
    need to know before it commits to anything - `DV_Service --tick` refuses at startup rather
    than beating forever on a workstation that could never answer."""
    try:
        return importlib.import_module('blpapi')
    except ImportError as error:
        raise BloombergUnavailable(
            'Bloomberg blpapi is unavailable. Install the Bloomberg-supported Python SDK on '
            'this workstation and confirm the Desktop API service is running.') from error


def _error_text(element) -> str:
    return str(element).strip()


def _scalar_value(element):
    """One field element as the reader has always read it - `getValue()`, element zero."""
    return element.getValue()


def _bulk_value(element):
    """One field element as a BULK field: the list of rows it carries, each row a dict of its own
    sub-fields, and a plain `getValue()` where the field is not an array after all.

    `getValue()` ALONE IS A SILENT TRUNCATION HERE, which is the whole reason this exists: on an
    array element it returns value ZERO, so `OPT_CHAIN` - two thousand listed contracts - arrives
    as one opaque row with no error anywhere. A reader that cannot see the shape of what it read is
    the dead-benchmark trap in another costume.
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
    """Small synchronous wrapper over Bloomberg Desktop API reference data."""

    def __init__(self, host: str = 'localhost', port: int = 8194, timeout_ms: int = 10000,
                 connect_timeout_ms: int = None):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        #: The whole budget for GETTING connected, which `timeout_ms` does not touch - that one
        #: bounds `nextEvent`, the request. A terminal that is not there spends its time in the
        #: socket and in the service handshake instead, and the SDK's defaults there are generous
        #: on purpose (5s x 3 attempts, then a minute of service checks). A caller that named a
        #: budget meant the whole of it, so this caps every leg of it at once. None leaves the
        #: SDK's own defaults, so every existing caller is untouched.
        self.connect_timeout_ms = connect_timeout_ms
        self._api = None
        self._session = None
        self._service = None

    def start(self):
        session = None
        try:
            api = blpapi_module()
            options = api.SessionOptions()
            options.setServerHost(self.host)
            options.setServerPort(self.port)
            if self.connect_timeout_ms is not None:
                # one attempt, not the SDK's three: a per-attempt timeout the library then
                # multiplies (and backs off between) is not a budget, it is a suggestion
                options.setConnectTimeout(self.connect_timeout_ms)
                options.setNumStartAttempts(1)
                options.setServiceCheckTimeout(self.connect_timeout_ms)
                options.setServiceDownloadTimeout(self.connect_timeout_ms)
            session = api.Session(options)
            if not session.start():
                raise BloombergUnavailable(
                    'Bloomberg Desktop API session did not start at {}:{}'.format(self.host, self.port))
            if not session.openService('//blp/refdata'):
                raise BloombergUnavailable('Bloomberg service //blp/refdata could not be opened')
            self._api = api
            self._session = session
            self._service = session.getService('//blp/refdata')
            return self
        except BloombergUnavailable:
            if session is not None:
                session.stop()
            raise
        except Exception as error:
            if session is not None:
                session.stop()
            raise BloombergUnavailable('Bloomberg Desktop API session failed: {}'.format(error)) from error

    def stop(self) -> None:
        if self._session is not None:
            self._session.stop()
        self._api = self._session = self._service = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def reference_data(self, securities: Sequence[str],
                       fields: Sequence[str]) -> dict[str, dict[str, object]]:
        """`{security: {field: value}}`, refusing the whole batch on ANY per-security error: a
        production tick built from a partial answer is a wrong market, not a smaller one."""
        response = {}
        for security, error, values in self._walked(securities, fields):
            if error is not None:
                raise_response_error('{}: {}'.format(security, error))
            response[security] = values
        return response

    def reference_data_report(self, securities: Sequence[str],
                              fields: Sequence[str]) -> dict[str, dict[str, object]]:
        """Per-security outcomes, for DISCOVERY: `{security: {'ok', 'error', 'fields'}}` with
        every requested name answered. One bad ticker in a batch of fifty is the finding there,
        not a failure - `reference_data` above is the production reader. A request-level error
        (a timeout, a `responseError`) still raises: that is transport, not a fact about a name."""
        return self._reported(securities, self._walked(securities, fields))

    def bulk_reference_data_report(self, securities: Sequence[str],
                                   fields: Sequence[str]) -> dict[str, dict[str, object]]:
        """`reference_data_report`'s contract over BULK fields: same `{security: {'ok', 'error',
        'fields'}}`, same tolerance, but each field answers a LIST OF ROWS rather than one value.

        A SECOND READER RATHER THAN A FLAG ON THE FIRST, because the two differ in what a FIELD IS
        and not in policy - and because the scalar reader's `getValue()` cannot be made to answer
        `OPT_CHAIN` without truncating it to one row in silence (see `_bulk_value`). The walk itself
        is shared: `_walk` and `_walk_bulk` are one `_request` under two extractors, so a change to
        the event loop cannot land on one reader and miss the other. `discover.probe` batches this
        one unchanged, since the contract it batches against is the same.
        """
        return self._reported(securities, self._walked_bulk(securities, fields))

    @staticmethod
    def _reported(securities, walked):
        report = {}
        for security, error, values in walked:
            report[security] = {'ok': error is None, 'error': error, 'fields': values}
        for security in securities:
            report.setdefault(security, {'ok': False, 'error': 'no answer in the response',
                                         'fields': {}})
        return report

    def _walked(self, securities, fields):
        """The one event walk the SCALAR readers share, materialized so the wrapping below covers
        the whole response: `(security, error, values)` per name, `error` carrying Bloomberg's own
        text where it refused one. Materializing means the response is DRAINED before either
        policy raises - deliberately: the strict reader used to abandon the event loop
        mid-response, leaving the session dirty for its next request. The cost is that a
        transport failure on a later event outranks a per-security error already walked.

        The bulk reader walks the SAME `_request` under its own extractor, and this signature is
        left alone on purpose: the gates drive a session by overriding `_walk` with canned rows."""
        return self._drained(lambda: self._walk(securities, fields))

    def _walked_bulk(self, securities, fields):
        """`_walked` over the bulk extractor - the same drain, the same wrapping, the same
        precedence, so the two readers cannot come to differ in how a failure is typed."""
        return self._drained(lambda: self._walk_bulk(securities, fields))

    def _drained(self, walk):
        if self._session is None or self._service is None or self._api is None:
            raise BloombergUnavailable('BloombergSession must be started before requesting data')
        try:
            return list(walk())
        except (BloombergRequestError, BloombergUnavailable):
            raise
        except Exception as error:
            raise BloombergRequestError('Bloomberg reference-data request failed: {}'.format(error)) from error

    def _walk(self, securities, fields):
        yield from self._request(securities, fields, _scalar_value)

    def _walk_bulk(self, securities, fields):
        yield from self._request(securities, fields, _bulk_value)

    def _request(self, securities, fields, value_of):
        request = self._service.createRequest('ReferenceDataRequest')
        security_element = request.getElement('securities')
        field_element = request.getElement('fields')
        for security in securities:
            security_element.appendValue(security)
        for field in fields:
            field_element.appendValue(field)
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