# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for HTTP certificate fallback handling and distrusted-CA hardening."""

import base64
import datetime
import hashlib
import re
import socket
import ssl
import textwrap
import threading
import urllib.error
from pathlib import Path

import pytest

# Certificates of the eight eMudhra intermediates and their Sectigo cross-signs, from
# https://emudhra.we-have-a-problem.com/impacted-intermediates.json
COMPROMISED_INTERMEDIATE_SHA256 = {
    "42c4accd12c4f59597925e1a08a7a90102608543809308d50ec427e118ba1493",
    "45a883f349aedfd5f3a649b5498ffb121712ba8b448f3f2a91dbc5a766fcddbe",
    "98b896625c0c230adeec3eb2cf6aad5001ecf9156a68644e82374ce5833bc25b",
    "5f1ded4dd590af74631290cf073b122a2aac6a1fa479778a0bd43579bd832699",
    "e7568fe6e12331f4fd7de61da4c7559d0f8eb192a218d4741a3ba75184333838",
    "579480aea166290bd3c643589e795fc0d0e54e86505d180f1dff99efd3199512",
    "0164d7ceae5bb909c272af151ff44b1177ae9b47d9424b94f1d9558c30934d19",
    "62b7f9868263f72f6911fc78ceaf72c1979158b7368bf648683bb2aa39b02e72",
    "cb6e15165ac802e9b3f993f16c25d41f4e81c21ebde71ae1b012ac00bf9c7c3f",
    "7485b51c0ed192878fa1d2f948fb529ae1842cc5086c4b67f7be5b2039de9489",
    "432ec25f0d28de9006d97bf752f17bbb2cd41f669933fcb072d9a1148d0261f3",
    "fd9d5c2dbbd4995ab69b8d50c439d9527b00529199c2fd01b2174eb58f07021f",
    "9e39de2881273acc225a149c72ba3e0f51424963669ab633ee1dc7df5db067b4",
    "a496e23ac0cab8155fc9b117b97eae368efcbb9041aec8804d5a308d2d01c2d8",
    "4b2460a466f95fd40113bcdcca2ceb29b3b093921de45474da7df48874c630b1",
    "84112972b8e59d14560ae7e7119654474aad119990cf3722bfc1e3b91615e1fd",
}

# X509_CERT_AUX holding only reject = { id-kp-serverAuth }.
REJECT_SERVER_AUTH_AUX = bytes.fromhex("300ca00a06082b06010505070301")


class _ResponseStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"ok"


def test_urlopen_retries_with_fallback_context(monkeypatch):
    from lfs_plugins import http

    system_context = object()
    fallback_context = object()
    calls = []

    monkeypatch.setattr(http, "_system_ssl_context", lambda: system_context)
    monkeypatch.setattr(http, "_fallback_ssl_context", lambda: fallback_context)

    cert_error = urllib.error.URLError(
        ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED")
    )

    def fake_urlopen(url, timeout=None, **kwargs):
        calls.append(kwargs.get("context"))
        if len(calls) == 1:
            raise cert_error
        return _ResponseStub()

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)

    resp = http.urlopen("https://example.com", timeout=5)

    assert isinstance(resp, _ResponseStub)
    assert calls == [system_context, fallback_context]


def test_urlopen_does_not_retry_non_certificate_errors(monkeypatch):
    from lfs_plugins import http

    system_context = object()
    calls = []

    monkeypatch.setattr(http, "_system_ssl_context", lambda: system_context)
    monkeypatch.setattr(http, "_fallback_ssl_context", lambda: object())

    def fake_urlopen(url, timeout=None, **kwargs):
        calls.append(kwargs.get("context"))
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError, match="connection refused"):
        http.urlopen("https://example.com", timeout=5)

    assert calls == [system_context]


def test_urlopen_propagates_cert_error_without_fallback_context(monkeypatch):
    from lfs_plugins import http

    system_context = object()
    calls = []
    cert_error = urllib.error.URLError(
        ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED")
    )

    monkeypatch.setattr(http, "_system_ssl_context", lambda: system_context)
    monkeypatch.setattr(http, "_fallback_ssl_context", lambda: None)

    def fake_urlopen(url, timeout=None, **kwargs):
        calls.append(kwargs.get("context"))
        raise cert_error

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        http.urlopen("https://example.com", timeout=5)

    assert calls == [system_context]


def test_distrusted_bundle_rejects_every_compromised_intermediate_for_server_auth():
    from lfs_plugins import http

    text = Path(http._DISTRUSTED_CAFILE).read_text(encoding="ascii")
    blocks = re.findall(
        r"-----BEGIN TRUSTED CERTIFICATE-----(.*?)-----END TRUSTED CERTIFICATE-----", text, re.S
    )
    fingerprints = set()
    for block in blocks:
        der = base64.b64decode("".join(block.split()))
        length_bytes = der[1] & 0x7F
        cert_end = 2 + length_bytes + int.from_bytes(der[2 : 2 + length_bytes], "big")
        fingerprints.add(hashlib.sha256(der[:cert_end]).hexdigest())
        assert der[cert_end:] == REJECT_SERVER_AUTH_AUX

    assert len(blocks) == len(COMPROMISED_INTERMEDIATE_SHA256)
    assert fingerprints == COMPROMISED_INTERMEDIATE_SHA256


@pytest.fixture(scope="module")
def tls_chain(tmp_path_factory):
    """Write a throwaway root -> distrusted intermediate -> leaf chain for lichtfeld.test."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)

    def issue(common_name, key, issuer_cert, issuer_key, *, ca, path_length=None):
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer_cert.subject if issuer_cert else subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=path_length), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=not ca, content_commitment=False, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=ca, crl_sign=ca,
                    encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        )
        if issuer_cert:
            builder = builder.add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False
            )
        if not ca:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.DNSName("lichtfeld.test")]), critical=False
            ).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        return builder.sign(issuer_key, hashes.SHA256())

    root_key, intermediate_key, leaf_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = issue("LichtFeld Test Root", root_key, None, root_key, ca=True)
    intermediate = issue("LichtFeld Test Distrusted Intermediate", intermediate_key, root, root_key,
                         ca=True, path_length=0)
    leaf = issue("lichtfeld.test", leaf_key, intermediate, intermediate_key, ca=False)

    def pem(cert):
        return cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    trusted = base64.b64encode(intermediate.public_bytes(serialization.Encoding.DER) + REJECT_SERVER_AUTH_AUX)
    directory = tmp_path_factory.mktemp("tls_distrust")
    (directory / "root.pem").write_text(pem(root))
    (directory / "intermediate.pem").write_text(pem(intermediate))
    (directory / "intermediate-rejected.pem").write_text(
        "-----BEGIN TRUSTED CERTIFICATE-----\n"
        + "\n".join(textwrap.wrap(trusted.decode("ascii"), 64))
        + "\n-----END TRUSTED CERTIFICATE-----\n"
    )
    (directory / "leaf-chain.pem").write_text(pem(leaf) + pem(intermediate))
    (directory / "leaf.key").write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return directory


def _handshake(client_context, chain):
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(chain / "leaf-chain.pem", chain / "leaf.key")
    server_sock, client_sock = socket.socketpair()
    server_sock.settimeout(10)
    client_sock.settimeout(10)

    def serve():
        try:
            with server_context.wrap_socket(server_sock, server_side=True):
                pass
        except (ssl.SSLError, OSError):
            pass
        finally:
            server_sock.close()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        with client_context.wrap_socket(client_sock, server_hostname="lichtfeld.test"):
            pass
    finally:
        client_sock.close()
        server.join(timeout=10)


def test_generated_chain_verifies_without_distrust(tls_chain):
    _handshake(ssl.create_default_context(cafile=tls_chain / "root.pem"), tls_chain)


# Loading the OS store before the distrusted CAs would let the ordinary copy of the
# intermediate win, so the handshakes below would succeed.
def test_system_context_refuses_distrusted_intermediate_listed_by_the_os(monkeypatch, tls_chain):
    from lfs_plugins import http

    def load_os_store_with_plain_intermediate(context, purpose=ssl.Purpose.SERVER_AUTH):
        """Stand-in for Windows, whose "CA" store can list a distrusted intermediate as an ordinary CA."""
        context.load_verify_locations(cafile=tls_chain / "root.pem")
        context.load_verify_locations(cafile=tls_chain / "intermediate.pem")

    monkeypatch.setattr(http, "_DISTRUSTED_CAFILE", str(tls_chain / "intermediate-rejected.pem"))
    monkeypatch.setattr(ssl.SSLContext, "load_default_certs", load_os_store_with_plain_intermediate)

    with pytest.raises(ssl.SSLCertVerificationError, match="certificate rejected"):
        _handshake(http._system_ssl_context.__wrapped__(), tls_chain)


def test_fallback_context_refuses_distrusted_intermediate_in_ca_bundle(monkeypatch, tmp_path, tls_chain):
    from lfs_plugins import http

    bundle = tmp_path / "cacert.pem"
    bundle.write_text((tls_chain / "root.pem").read_text() + (tls_chain / "intermediate.pem").read_text())
    monkeypatch.setattr(http, "_DISTRUSTED_CAFILE", str(tls_chain / "intermediate-rejected.pem"))
    monkeypatch.setattr(http, "_load_ca_bundle_from_certifi", lambda: str(bundle))

    with pytest.raises(ssl.SSLCertVerificationError, match="certificate rejected"):
        _handshake(http._fallback_ssl_context.__wrapped__(), tls_chain)
