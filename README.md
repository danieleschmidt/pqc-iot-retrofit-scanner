# PQC IoT Retrofit Scanner

A CLI tool that scans IoT firmware, embedded C/C++, Python, and configuration
files for quantum-vulnerable cryptographic usage and recommends NIST-standardized
post-quantum replacements.

```
$ pqc-scan ./my-firmware/
Scanning ./my-firmware/ …
Done. 47 files, 12 findings.

========================================================================
  PQC IoT Retrofit Scanner — Results
========================================================================
  Target       : ./my-firmware/
  Files scanned: 47
  Total findings: 12

  By severity:
    CRITICAL   4
    HIGH       7
    MEDIUM     1

  ./my-firmware/src/tls.c
  -------------------------
    [CRITICAL] line:23  RSA
               detail : OpenSSL RSA API
               context: RSA *key = RSA_new();
               replace: ML-KEM-768 (NIST FIPS 203) for key exchange; ML-DSA-65 (NIST FIPS 204) for signatures
```

---

## Why This Exists

A sufficiently powerful quantum computer running Shor's algorithm will break:

- **RSA** (all key sizes — just a matter of qubit count)
- **ECC / ECDSA / ECDH** (all curves)
- **DH / DHE** (finite-field Diffie-Hellman)
- **DSA** (classic discrete-log signatures)

NIST finalized three post-quantum standards in 2024. IoT devices shipped today
will still be operating when fault-tolerant quantum computers arrive. This tool
helps you find the vulnerable code now, before it's too late to retrofit.

---

## NIST PQC Standards (2024)

| Standard    | Algorithm              | Purpose                     | Replaces              |
|-------------|------------------------|-----------------------------|-----------------------|
| FIPS 203    | **ML-KEM** (Kyber)     | Key encapsulation / KEM     | RSA-KEM, ECDH, DH     |
| FIPS 204    | **ML-DSA** (Dilithium) | Digital signatures          | RSA-sign, ECDSA, DSA  |
| FIPS 205    | **SLH-DSA** (SPHINCS+) | Stateless hash-based sigs   | Backup signature alt  |

Grover's algorithm weakens symmetric crypto (AES, SHA) by halving the effective
key/output size. Mitigation: use **AES-256** and **SHA-256** minimum.

### Quantum Threat Timeline

| Year (est.) | Event |
|-------------|-------|
| 2024        | NIST finalizes ML-KEM, ML-DSA, SLH-DSA |
| 2027–2030   | Cryptographically relevant quantum computers (CRQCs) possible |
| 2030+       | RSA-2048 / ECC-256 no longer safe per CISA/CNSA 2.0 guidance |

CISA CNSA 2.0 suite mandates PQC-only for national security systems by 2033.

---

## Installation

```bash
pip install -e .
# or
pip install pqc-iot-retrofit-scanner
```

Requires Python 3.8+. No heavy dependencies — only `click`.

---

## Usage

### Scan a directory
```bash
pqc-scan /path/to/firmware/
```

### JSON output
```bash
pqc-scan /path/to/firmware/ --format json
pqc-scan /path/to/firmware/ --format json --output report.json
```

### Filter by severity
```bash
pqc-scan /path/to/firmware/ --min-severity HIGH
```

### CI integration (fail on CRITICAL findings)
```bash
pqc-scan /path/to/firmware/ --fail-on CRITICAL
echo $?  # 1 if CRITICAL findings found
```

### Exclude directories
```bash
pqc-scan /path/to/firmware/ --exclude vendor --exclude third_party
```

---

## What It Detects

### C/C++ Source
- OpenSSL: `RSA_new`, `RSA_generate_key`, `ECDSA_sign`, `ECDH_compute_key`, `DH_new`, `DSA_new`, …
- mbedTLS: `mbedtls_rsa_init`, `mbedtls_ecdsa_sign`, `mbedtls_dhm_make_params`, …
- wolfSSL: `wolfSSL_RSA_new`, `wc_ecc_sign_hash`, `wc_DhGenerateKeyPair`, …

### Python
- `cryptography` library: `from cryptography.hazmat.primitives.asymmetric import rsa/ec/dh/dsa`
- PyCryptodome/PyCryptodomex: `from Crypto.PublicKey import RSA/ECC/DSA`
- `rsa` package: `import rsa`
- `paramiko` quantum-vulnerable key types

### Configuration Files (`.conf`, `.cfg`, `.yaml`, `.ini`, …)
- TLS cipher suites containing RSA/ECDSA/DHE
- Deprecated TLS versions (TLSv1.0, TLSv1.1, SSLv2/3)
- SSH host key type configuration
- Key algorithm settings

### Certificates & Keys (`.pem`, `.crt`, `.key`, `.der`, …)
- PEM headers: `BEGIN RSA PRIVATE KEY`, `BEGIN EC PRIVATE KEY`, `BEGIN DSA PRIVATE KEY`
- DER-encoded ASN.1 OIDs for RSA, EC, DSA, DH

### Binary Firmware (`.bin`, `.elf`, `.hex`, `.img`, `.fw`, …)
- ASN.1 OID byte sequences embedded in binary blobs
- PEM header strings in firmware flash regions

---

## Risk Scoring

| Severity | Meaning |
|----------|---------|
| CRITICAL | Quantum-broken, high IoT deployment prevalence (RSA, DSA) |
| HIGH     | Quantum-broken, moderate deployment (ECDSA, ECDH, DH) |
| MEDIUM   | Quantum-weakened or classically-weak (MD5, SHA-1, AES-128) |
| INFO     | Informational / best-practice flags |

---

## JSON Output Schema

```json
{
  "target": "/path/to/scan",
  "scanned_at": "2024-11-01T12:00:00+00:00",
  "files_scanned": 47,
  "summary": {
    "total_findings": 12,
    "by_severity": {
      "CRITICAL": 4,
      "HIGH": 7,
      "MEDIUM": 1,
      "INFO": 0
    }
  },
  "findings": [
    {
      "file": "src/tls.c",
      "line": 23,
      "byte_offset": null,
      "algorithm": "RSA",
      "category": "RSA",
      "severity": "CRITICAL",
      "detail": "OpenSSL RSA API",
      "pqc_replacement": "ML-KEM-768 (NIST FIPS 203) for key exchange; ML-DSA-65 (NIST FIPS 204) for signatures",
      "context": "RSA *key = RSA_new();"
    }
  ]
}
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## References

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [FIPS 203 — ML-KEM (Kyber)](https://csrc.nist.gov/pubs/fips/203/final)
- [FIPS 204 — ML-DSA (Dilithium)](https://csrc.nist.gov/pubs/fips/204/final)
- [FIPS 205 — SLH-DSA (SPHINCS+)](https://csrc.nist.gov/pubs/fips/205/final)
- [CISA CNSA 2.0 Suite](https://www.nsa.gov/Press-Room/News-Highlights/Article/Article/3148990/nsa-releases-future-quantum-resistant-qr-algorithm-requirements-for-national-se/)
- [liboqs — Open Quantum Safe](https://openquantumsafe.org/)

---

## License

MIT
