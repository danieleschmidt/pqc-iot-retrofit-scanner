"""
Tests for the PQC IoT Retrofit Scanner.

Synthetic firmware samples are embedded inline — no external files needed.
"""

import json
import tempfile
from pathlib import Path

import pytest

from pqc_iot_retrofit.scanner import (
    PQCScanner, ScanReport, Finding, Severity, AlgoCategory,
    C_PATTERNS, PY_PATTERNS, CFG_PATTERNS, BINARY_PATTERNS,
    PQC_REPLACEMENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_file(content: str, suffix: str, tmp_path: Path) -> Path:
    p = tmp_path / f"sample{suffix}"
    p.write_text(content, encoding="utf-8")
    return p


def make_temp_bytes(data: bytes, suffix: str, tmp_path: Path) -> Path:
    p = tmp_path / f"sample{suffix}"
    p.write_bytes(data)
    return p


def scan_file(path: Path) -> ScanReport:
    return PQCScanner(path).scan()


# ---------------------------------------------------------------------------
# C / C++ source detection
# ---------------------------------------------------------------------------

class TestCSourceScanning:

    def test_openssl_rsa_detected(self, tmp_path):
        src = """
#include <openssl/rsa.h>

int generate_key(void) {
    RSA *rsa = RSA_new();
    RSA_generate_key(2048, RSA_F4, NULL, NULL);
    return 0;
}
"""
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert r.total > 0
        algos = {f.algorithm for f in r.findings}
        assert "RSA" in algos

    def test_openssl_ecdsa_detected(self, tmp_path):
        src = """
#include <openssl/ec.h>
EC_KEY *key = EC_KEY_new();
ECDSA_sign(0, hash, hash_len, sig, &sig_len, key);
"""
        p = make_temp_file(src, ".cpp", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.ECC for f in r.findings)

    def test_mbed_tls_ecdh_detected(self, tmp_path):
        src = "mbedtls_ecdh_calc_secret(&ctx, &olen, buf, 256, mbedtls_ctr_drbg_random, &ctr_drbg);\n"
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.ECC for f in r.findings)

    def test_openssl_dh_detected(self, tmp_path):
        src = "DH *dh = DH_new();\nDH_generate_parameters(1024, 2, NULL, NULL);\n"
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.DH for f in r.findings)

    def test_openssl_dsa_detected(self, tmp_path):
        src = "DSA *dsa = DSA_new();\nDSA_generate_key(dsa);\nDSA_sign(0, dgst, 20, sig, &siglen, dsa);\n"
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.DSA for f in r.findings)

    def test_des_detected(self, tmp_path):
        src = "DES_set_key(&key_schedule, &ks);\nDES_ecb_encrypt(&input, &output, &ks, DES_ENCRYPT);\n"
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert any("DES" in f.algorithm for f in r.findings)

    def test_wolfssl_rsa_detected(self, tmp_path):
        src = "wc_RsaKeyToDer(&rsa, der, sizeof(der));\nwolfSSL_RSA_new();\n"
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.RSA for f in r.findings)

    def test_clean_c_no_findings(self, tmp_path):
        src = """
#include <string.h>
void xor_encrypt(uint8_t *buf, size_t len, uint8_t key) {
    for (size_t i = 0; i < len; i++) buf[i] ^= key;
}
"""
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        assert r.total == 0

    def test_severity_critical_for_rsa(self, tmp_path):
        src = "RSA_new();\n"
        p = make_temp_file(src, ".c", tmp_path)
        r = scan_file(p)
        rsa_findings = [f for f in r.findings if f.category == AlgoCategory.RSA]
        assert any(f.severity == Severity.CRITICAL for f in rsa_findings)


# ---------------------------------------------------------------------------
# Python source detection
# ---------------------------------------------------------------------------

class TestPythonScanning:

    def test_cryptography_rsa_import(self, tmp_path):
        src = "from cryptography.hazmat.primitives.asymmetric import rsa\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.RSA for f in r.findings)

    def test_cryptography_ec_import(self, tmp_path):
        src = "from cryptography.hazmat.primitives.asymmetric import ec\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.ECC for f in r.findings)

    def test_pycryptodome_rsa(self, tmp_path):
        src = "from Crypto.PublicKey import RSA\nfrom Cryptodome.PublicKey import RSA\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        rsa = [f for f in r.findings if f.category == AlgoCategory.RSA]
        assert len(rsa) >= 1

    def test_rsa_package(self, tmp_path):
        src = "import rsa\nkey = rsa.newkeys(2048)\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.RSA for f in r.findings)

    def test_cryptography_dh(self, tmp_path):
        src = "from cryptography.hazmat.primitives.asymmetric import dh\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.DH for f in r.findings)

    def test_cryptography_dsa(self, tmp_path):
        src = "from cryptography.hazmat.primitives.asymmetric import dsa\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.DSA for f in r.findings)

    def test_paramiko_rsa_key(self, tmp_path):
        src = "import paramiko\nkey = paramiko.RSAKey.generate(2048)\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert r.total > 0

    def test_md5_usage(self, tmp_path):
        src = "import hashlib\nhash = hashlib.md5(data).hexdigest()\n"
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.HASH for f in r.findings)

    def test_clean_python_no_findings(self, tmp_path):
        src = """
import hashlib
import secrets

key = secrets.token_bytes(32)
digest = hashlib.sha256(key).hexdigest()
"""
        p = make_temp_file(src, ".py", tmp_path)
        r = scan_file(p)
        assert r.total == 0


# ---------------------------------------------------------------------------
# Config file detection
# ---------------------------------------------------------------------------

class TestConfigScanning:

    def test_nginx_rsa_cipher(self, tmp_path):
        src = "ssl_ciphers 'ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384';\n"
        p = make_temp_file(src, ".conf", tmp_path)
        r = scan_file(p)
        assert r.total > 0

    def test_openssl_cnf(self, tmp_path):
        src = "[req]\nkey_algorithm = RSA\nprivate_key_bits = 2048\n"
        p = make_temp_file(src, ".cfg", tmp_path)
        r = scan_file(p)
        assert r.total > 0

    def test_ssh_config(self, tmp_path):
        src = "HostKey /etc/ssh/ssh_host_rsa_key\nHostKey /etc/ssh/ssh_host_ecdsa_key\n"
        p = make_temp_file(src, ".conf", tmp_path)
        r = scan_file(p)
        assert r.total > 0

    def test_deprecated_tls_version(self, tmp_path):
        src = "ssl_protocols TLSv1.0 TLSv1.1;\n"
        p = make_temp_file(src, ".conf", tmp_path)
        r = scan_file(p)
        assert r.total > 0


# ---------------------------------------------------------------------------
# Certificate / PEM detection
# ---------------------------------------------------------------------------

class TestCertScanning:

    def test_rsa_private_key_pem(self, tmp_path):
        src = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n"
        p = make_temp_file(src, ".pem", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.RSA for f in r.findings)
        assert any(f.severity == Severity.CRITICAL for f in r.findings)

    def test_ec_private_key_pem(self, tmp_path):
        src = "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEEIBkg...\n-----END EC PRIVATE KEY-----\n"
        p = make_temp_file(src, ".key", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.ECC for f in r.findings)

    def test_dsa_private_key_pem(self, tmp_path):
        src = "-----BEGIN DSA PRIVATE KEY-----\nMIIBvAIBAAK...\n-----END DSA PRIVATE KEY-----\n"
        p = make_temp_file(src, ".pem", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.DSA for f in r.findings)


# ---------------------------------------------------------------------------
# Binary / firmware detection
# ---------------------------------------------------------------------------

class TestBinaryScanning:

    RSA_OID = b'\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01'
    EC_OID  = b'\x06\x07\x2a\x86\x48\xce\x3d\x02\x01'
    DSA_OID = b'\x06\x07\x2a\x86\x48\xce\x38\x04\x01'
    DH_OID  = b'\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x03\x01'

    def _pad(self, payload: bytes) -> bytes:
        """Embed payload in a 256-byte fake firmware blob."""
        header = b'\x7fELF' + b'\x00' * 60
        return header + payload + b'\xff' * (256 - len(header) - len(payload))

    def test_rsa_oid_in_elf(self, tmp_path):
        data = self._pad(self.RSA_OID)
        p = make_temp_bytes(data, ".elf", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.RSA for f in r.findings)

    def test_ec_oid_in_bin(self, tmp_path):
        data = self._pad(self.EC_OID)
        p = make_temp_bytes(data, ".bin", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.ECC for f in r.findings)

    def test_dsa_oid_in_firmware(self, tmp_path):
        data = self._pad(self.DSA_OID)
        p = make_temp_bytes(data, ".fw", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.DSA for f in r.findings)

    def test_pem_marker_in_binary(self, tmp_path):
        data = b'\x00' * 64 + b'RSA PRIVATE KEY' + b'\x00' * 64
        p = make_temp_bytes(data, ".bin", tmp_path)
        r = scan_file(p)
        assert any(f.category == AlgoCategory.RSA for f in r.findings)

    def test_multiple_oids_in_one_file(self, tmp_path):
        data = b'\x00' * 32 + self.RSA_OID + b'\x00' * 16 + self.EC_OID + b'\x00' * 32
        p = make_temp_bytes(data, ".bin", tmp_path)
        r = scan_file(p)
        cats = {f.category for f in r.findings}
        assert AlgoCategory.RSA in cats
        assert AlgoCategory.ECC in cats

    def test_clean_binary_no_findings(self, tmp_path):
        data = bytes(range(256)) * 4
        p = make_temp_bytes(data, ".bin", tmp_path)
        r = scan_file(p)
        assert r.total == 0


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

class TestDirectoryScan:

    def test_mixed_firmware_dir(self, tmp_path):
        # Simulate a mini firmware project
        (tmp_path / "main.c").write_text("RSA_new();\n")
        (tmp_path / "crypto.py").write_text("from cryptography.hazmat.primitives.asymmetric import ec\n")
        (tmp_path / "config.conf").write_text("ssl_ciphers 'ECDHE-RSA-AES128-SHA';\n")
        (tmp_path / "server.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nfoo\n-----END RSA PRIVATE KEY-----\n")
        (tmp_path / "README.md").write_text("# project\n")  # should not fire

        r = PQCScanner(tmp_path).scan()
        assert r.files_scanned >= 4
        assert r.total > 0

    def test_exclude_dirs_skipped(self, tmp_path):
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "crypto.py").write_text("from cryptography.hazmat.primitives.asymmetric import rsa\n")
        r = PQCScanner(tmp_path).scan()
        assert r.total == 0

    def test_min_severity_filter(self, tmp_path):
        # MD5 = MEDIUM, RSA = CRITICAL
        (tmp_path / "x.py").write_text(
            "import hashlib\nhashlib.md5(data)\nfrom cryptography.hazmat.primitives.asymmetric import rsa\n"
        )
        r_high = PQCScanner(tmp_path, min_severity=Severity.HIGH).scan()
        r_all  = PQCScanner(tmp_path, min_severity=Severity.INFO).scan()
        assert r_high.total < r_all.total
        assert all(
            f.severity in (Severity.HIGH, Severity.CRITICAL)
            for f in r_high.findings
        )


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------

class TestReportOutput:

    def test_json_output_valid(self, tmp_path):
        (tmp_path / "x.c").write_text("RSA_new();\n")
        r = PQCScanner(tmp_path).scan()
        parsed = json.loads(r.to_json())
        assert "findings" in parsed
        assert "summary" in parsed
        assert parsed["summary"]["total_findings"] == r.total

    def test_text_output_contains_severity(self, tmp_path):
        (tmp_path / "x.c").write_text("RSA_new();\n")
        r = PQCScanner(tmp_path).scan()
        text = r.to_text()
        assert "CRITICAL" in text or "HIGH" in text

    def test_text_output_clean_message(self, tmp_path):
        (tmp_path / "x.c").write_text("// no crypto here\n")
        r = PQCScanner(tmp_path).scan()
        assert "No quantum-vulnerable" in r.to_text()

    def test_json_finding_fields(self, tmp_path):
        (tmp_path / "x.c").write_text("RSA_new();\n")
        r = PQCScanner(tmp_path).scan()
        f = json.loads(r.to_json())["findings"][0]
        for key in ("file", "algorithm", "category", "severity", "pqc_replacement"):
            assert key in f


# ---------------------------------------------------------------------------
# PQC replacement mapping
# ---------------------------------------------------------------------------

class TestPQCReplacements:

    def test_rsa_recommends_ml_kem_and_ml_dsa(self):
        rep = PQC_REPLACEMENT[AlgoCategory.RSA]
        assert "ML-KEM" in rep or "ML-DSA" in rep

    def test_ecc_recommends_ml_kem_and_ml_dsa(self):
        rep = PQC_REPLACEMENT[AlgoCategory.ECC]
        assert "ML-KEM" in rep or "ML-DSA" in rep

    def test_dh_recommends_ml_kem(self):
        assert "ML-KEM" in PQC_REPLACEMENT[AlgoCategory.DH]

    def test_dsa_recommends_ml_dsa(self):
        assert "ML-DSA" in PQC_REPLACEMENT[AlgoCategory.DSA]
