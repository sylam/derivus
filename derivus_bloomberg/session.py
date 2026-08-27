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

from .errors import BloombergEntitlementError, BloombergRequestError, BloombergUnavailable


def _blpapi():
    try:
        return importlib.import_module('blpapi')
    except ImportError as error:
        raise BloombergUnavailable(
            'Bloomberg blpapi is unavailable. Install the Bloomberg-supported Python SDK on '
            'this workstation and confirm the Desktop API service is running.') from error


def _error_text(element) -> str:
    return str(element).strip()


def _raise_response_error(text: str) -> None:
    error_type = BloombergEntitlementError if any(
        token in text.upper() for token in ('NOT_AUTHORIZED', 'NOT_ENTITLED', 'NO_AUTH')) \
        else BloombergRequestError
    raise error_type(text)


class BloombergSession:
    """Small synchronous wrapper over Bloomberg Desktop API reference data."""

    def __init__(self, host: str = 'localhost', port: int = 8194, timeout_ms: int = 10000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._api = None
        self._session = None
        self._service = None

    def start(self):
        session = None
        try:
            api = _blpapi()
            options = api.SessionOptions()
            options.setServerHost(self.host)
            options.setServerPort(self.port)
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
                _raise_response_error('{}: {}'.format(security, error))
            response[security] = values
        return response

    def reference_data_report(self, securities: Sequence[str],
                              fields: Sequence[str]) -> dict[str, dict[str, object]]:
        """Per-security outcomes, for DISCOVERY: `{security: {'ok', 'error', 'fields'}}` with
        every requested name answered. One bad ticker in a batch of fifty is the finding there,
        not a failure - `reference_data` above is the production reader. A request-level error
        (a timeout, a `responseError`) still raises: that is transport, not a fact about a name."""
        report = {}
        for security, error, values in self._walked(securities, fields):
            report[security] = {'ok': error is None, 'error': error, 'fields': values}
        for security in securities:
            report.setdefault(security, {'ok': False, 'error': 'no answer in the response',
                                         'fields': {}})
        return report

    def _walked(self, securities, fields):
        """The one event walk both readers share, materialized so the wrapping below covers the
        whole response: `(security, error, values)` per name, `error` carrying Bloomberg's own
        text where it refused one. Materializing means the response is DRAINED before either
        policy raises - deliberately: the strict reader used to abandon the event loop
        mid-response, leaving the session dirty for its next request. The cost is that a
        transport failure on a later event outranks a per-security error already walked."""
        if self._session is None or self._service is None or self._api is None:
            raise BloombergUnavailable('BloombergSession must be started before requesting data')
        try:
            return list(self._walk(securities, fields))
        except (BloombergRequestError, BloombergUnavailable):
            raise
        except Exception as error:
            raise BloombergRequestError('Bloomberg reference-data request failed: {}'.format(error)) from error

    def _walk(self, securities, fields):
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
                    _raise_response_error(_error_text(message.getElement('responseError')))
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
                        values = {field: data.getElement(field).getValue()
                                  for field in fields if data.hasElement(field)}
                    yield security, error, values
            if event.eventType() == self._api.Event.RESPONSE:
                return