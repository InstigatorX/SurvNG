"""Serialize real bundled ONVIF WSDLs without contacting any camera."""
from pathlib import Path

import onvif
import pytest
import zeep
from lxml import etree
from zeep.transports import Transport
from zeep.wsse.username import UsernameToken

from survng.app.onvif_events import (
    SUBSCRIPTION_MANAGER_BINDING, _SubscriptionManagerAddressing,
)

WSA = "http://www.w3.org/2005/08/addressing"
ENDPOINT = "http://fixture.invalid/onvif/subscription/123"


class OfflineTransport(Transport):
    def load(self, url):
        if url.startswith(("http://", "https://")):
            raise AssertionError("Test must not fetch remote WSDLs")
        return super().load(url)

    def post_xml(self, *args, **kwargs):
        raise AssertionError("Test must not send requests")


@pytest.fixture
def client():
    wsdl = Path(onvif.__file__).parent.parent / "wsdl" / "events.wsdl"
    transport = OfflineTransport()
    try:
        yield zeep.Client(
            str(wsdl), transport=transport, settings=zeep.Settings(strict=False),
            wsse=UsernameToken("fixture", "fixture", use_digest=True),
            plugins=[_SubscriptionManagerAddressing()],
        )
    finally:
        transport.session.close()


def envelope(client, binding, operation, **kwargs):
    service = client.create_service(binding, ENDPOINT)
    return service._binding._create(
        operation, (), kwargs, client=client, options=service._binding_options,
    )[0]


def check_addressing(xml, action):
    for name in ("Action", "To", "MessageID"):
        assert len(xml.findall(f".//{{{WSA}}}{name}")) == 1
    assert xml.findtext(f".//{{{WSA}}}Action") == action
    assert xml.findtext(f".//{{{WSA}}}To") == ENDPOINT
    assert xml.findtext(f".//{{{WSA}}}MessageID").startswith("urn:uuid:")
    assert len(xml.xpath("//*[local-name()='Security']")) == 1


@pytest.mark.parametrize("operation", ["Renew", "Unsubscribe"])
def test_manager_requests_receive_addressing_and_keep_security(client, operation):
    kwargs = {"TerminationTime": "PT1H"} if operation == "Renew" else {}
    xml = envelope(client, SUBSCRIPTION_MANAGER_BINDING, operation, **kwargs)
    check_addressing(xml, "http://docs.oasis-open.org/wsn/bw-2/SubscriptionManager/" + operation + "Request")
    if operation == "Renew":
        assert xml.xpath("//*[local-name()='TerminationTime']/text()") == ["PT1H"]
    second = envelope(client, SUBSCRIPTION_MANAGER_BINDING, operation, **kwargs)
    assert second.findtext(f".//{{{WSA}}}MessageID") != xml.findtext(f".//{{{WSA}}}MessageID")


@pytest.mark.parametrize("binding,operation,kwargs", [
    ("EventBinding", "CreatePullPointSubscription", {"InitialTerminationTime": "PT1H"}),
    ("PullPointSubscriptionBinding", "PullMessages", {"Timeout": "PT5S", "MessageLimit": 10}),
])
def test_automatically_addressed_operations_are_not_duplicated(client, binding, operation, kwargs):
    binding = "{http://www.onvif.org/ver10/events/wsdl}" + binding
    service = client.create_service(binding, ENDPOINT)
    expected_action = service._binding.get(operation).abstract.wsa_action
    assert expected_action
    check_addressing(envelope(client, binding, operation, **kwargs), expected_action)


def test_explicit_addressing_is_preserved_without_duplicate_headers(client):
    message_id = etree.Element(f"{{{WSA}}}MessageID")
    message_id.text = "urn:uuid:explicit-test-id"
    xml = envelope(client, SUBSCRIPTION_MANAGER_BINDING, "Renew",
                   TerminationTime="PT1H", _soapheaders=[message_id])
    check_addressing(xml, "http://docs.oasis-open.org/wsn/bw-2/SubscriptionManager/RenewRequest")
    assert xml.findtext(f".//{{{WSA}}}MessageID") == message_id.text
