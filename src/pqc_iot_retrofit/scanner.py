"""
PQC IoT Retrofit Scanner — Core Detection Engine

Detects quantum-vulnerable cryptographic usage across:
  - C/C++ source and headers (function calls, API patterns)
  - Python source (imports, library calls)
  - Configuration files (OpenSSL config, YAML, INI)
  - DER/PEM certificates and key files
  - Binary firmware blobs (byte-level pattern matching)

Quantum-vulnerable algorithms detected:
  RSA, DH/DHE, DSA, ECDSA, ECDH, EC (all curves)

PQC replacements recommended per NIST FIPS 203/204/205 (2024):
  ML-KEM (Kyber)     — key encapsulation (replaces RSA-KEM, ECDH, DH)
  ML-DSA (Dilithium) — digital signatures (replaces RSA-sign, ECDSA, DSA)
  SLH-DSA (SPHINCS+) — stateless hash-based signatures (backup option)
"""

from __future__ import annotations

import re
import json
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Iterator


# ---------------------------------------------------------------------------
# Algorithm taxonomy
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # quantum-broken, small key, widely deployed
    HIGH     = "HIGH"       # quantum-broken, larger key
    MEDIUM   = "MEDIUM"     # quantum-weakened (symmetric halved, hashes)
    INFO     = "INFO"       # deprecated but not directly quantum-broken


class AlgoCategory(str, Enum):
    RSA     = "RSA"
    ECC     = "ECC"       # ECDSA / ECDH
    DH      = "DH"        # finite-field DH / DHE
    DSA     = "DSA"       # classic DSA
    SYMM    = "SYMMETRIC" # AES-128 weakened; DES/3DES classical-broken
    HASH    = "HASH"      # MD5/SHA-1 classical-broken


# PQC replacement guidance
PQC_REPLACEMENT = {
    AlgoCategory.RSA:  "ML-KEM-768 (NIST FIPS 203) for key exchange; ML-DSA-65 (NIST FIPS 204) for signatures",
    AlgoCategory.ECC:  "ML-KEM-768 for ECDH; ML-DSA-65 for ECDSA",
    AlgoCategory.DH:   "ML-KEM-768 (NIST FIPS 203)",
    AlgoCategory.DSA:  "ML-DSA-65 (NIST FIPS 204) or SLH-DSA-SHAKE-128s (NIST FIPS 205)",
    AlgoCategory.SYMM: "AES-256 (quantum-safe), drop DES/3DES immediately",
    AlgoCategory.HASH:  "SHA-256 or SHA-3-256 minimum",
}


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    file: str
    line: Optional[int]           # None for binary matches
    byte_offset: Optional[int]    # None for text matches
    algorithm: str
    category: AlgoCategory
    severity: Severity
    detail: str                   # what matched
    pqc_replacement: str
    context: str = ""             # snippet of surrounding code/data

    def as_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d

    def text_line(self) -> str:
        loc = f":{self.line}" if self.line else f"+0x{self.byte_offset:x}" if self.byte_offset else ""
        return (
            f"[{self.severity.value:<8}] {self.file}{loc}  |  "
            f"{self.algorithm}  →  {self.pqc_replacement}"
        )


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# --- C/C++ function / API patterns ---
C_PATTERNS: list[tuple[re.Pattern, str, AlgoCategory, Severity, str]] = [
    # (pattern, algo_name, category, severity, detail)

    # RSA
    (re.compile(r'\bRSA_generate_key\b|\bRSA_new\b|\bRSA_private_encrypt\b|\bRSA_public_decrypt\b'),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "OpenSSL RSA API"),
    (re.compile(r'\bmbedtls_rsa_init\b|\bmbedtls_rsa_gen_key\b|\bmbedtls_rsa_pkcs1_encrypt\b'),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "mbedTLS RSA API"),
    (re.compile(r'\bwolfRSA_Init\b|\bwolfRSA_MakeKey\b|\bwolfSSL_RSA_new\b'),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "wolfSSL RSA API"),

    # ECDSA / ECDH
    (re.compile(r'\bEC_KEY_new\b|\bECDSA_sign\b|\bECDSA_verify\b|\bECDH_compute_key\b'),
     "ECDSA/ECDH", AlgoCategory.ECC, Severity.HIGH, "OpenSSL ECC API"),
    (re.compile(r'\bmbedtls_ecdsa_sign\b|\bmbedtls_ecdh_calc_secret\b|\bmbedtls_ecp_group_load\b'),
     "ECDSA/ECDH", AlgoCategory.ECC, Severity.HIGH, "mbedTLS ECC API"),
    (re.compile(r'\bwolfSSL_EC_KEY_new\b|\bwolfECC_init\b|\bwc_ecc_sign_hash\b|\bwc_ecc_shared_secret\b'),
     "ECDSA/ECDH", AlgoCategory.ECC, Severity.HIGH, "wolfSSL ECC API"),

    # DH
    (re.compile(r'\bDH_new\b|\bDH_generate_parameters\b|\bDHE_\w+'),
     "DH/DHE", AlgoCategory.DH, Severity.HIGH, "OpenSSL DH API"),
    (re.compile(r'\bmbedtls_dhm_init\b|\bmbedtls_dhm_make_params\b|\bmbedtls_dhm_calc_secret\b'),
     "DH/DHE", AlgoCategory.DH, Severity.HIGH, "mbedTLS DHM API"),
    (re.compile(r'\bwc_DhGenerateKeyPair\b|\bwc_DhCheckPubKey\b|\bwc_DhAgree\b'),
     "DH/DHE", AlgoCategory.DH, Severity.HIGH, "wolfSSL DH API"),

    # DSA
    (re.compile(r'\bDSA_new\b|\bDSA_sign\b|\bDSA_verify\b|\bDSA_generate_key\b'),
     "DSA", AlgoCategory.DSA, Severity.CRITICAL, "OpenSSL DSA API"),
    (re.compile(r'\bmbedtls_mpi_init\b.*\bmbedtls_mpi_exp_mod\b', re.DOTALL),
     "DSA", AlgoCategory.DSA, Severity.HIGH, "mbedTLS big-integer (possible DSA)"),

    # Deprecated symmetric / hashes (classical-broken, not quantum)
    (re.compile(r'\bDES_\w+|\bdes_\w+|\b3DES_\w+|\bDES3_\w+'),
     "DES/3DES", AlgoCategory.SYMM, Severity.CRITICAL, "DES/3DES (classically broken)"),
    (re.compile(r'\bMD5_Init\b|\bMD5_Update\b|\bMD5_Final\b|\bmbedtls_md5_\w+|\bwc_Md5\w+'),
     "MD5", AlgoCategory.HASH, Severity.MEDIUM, "MD5 (collision-broken)"),
    (re.compile(r'\bSHA1_Init\b|\bSHA1_Update\b|\bSHA_Final\b|\bmbedtls_sha1_\w+|\bwc_ShaHash\b'),
     "SHA-1", AlgoCategory.HASH, Severity.MEDIUM, "SHA-1 (collision-broken)"),
]

# --- Python import / usage patterns ---
PY_PATTERNS: list[tuple[re.Pattern, str, AlgoCategory, Severity, str]] = [
    # RSA
    (re.compile(r'from\s+cryptography\.hazmat\.primitives\.asymmetric\s+import\s+(?:[^#\n]*\b)?rsa\b'),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "cryptography.io RSA import"),
    (re.compile(r'from\s+Crypto(?:dome)?\.PublicKey\s+import\s+(?:[^#\n]*\b)?RSA\b'),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "PyCryptodome RSA import"),
    (re.compile(r'import\s+rsa\b'),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "rsa package import"),
    (re.compile(r'rsa\.newkeys\(|rsa\.encrypt\(|rsa\.decrypt\(|rsa\.sign\('),
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "rsa package usage"),

    # ECC
    (re.compile(r'from\s+cryptography\.hazmat\.primitives\.asymmetric\s+import\s+(?:[^#\n]*\b)?ec\b'),
     "ECDSA/ECDH", AlgoCategory.ECC, Severity.HIGH, "cryptography.io EC import"),
    (re.compile(r'from\s+Crypto(?:dome)?\.PublicKey\s+import\s+(?:[^#\n]*\b)?ECC\b'),
     "ECDSA/ECDH", AlgoCategory.ECC, Severity.HIGH, "PyCryptodome ECC import"),
    (re.compile(r'\bECDSA\b|\bECDH\b|\bec\.generate_private_key\(|\bec\.ECDH\('),
     "ECDSA/ECDH", AlgoCategory.ECC, Severity.HIGH, "ECC class/function usage"),

    # DH
    (re.compile(r'from\s+cryptography\.hazmat\.primitives\.asymmetric\s+import\s+(?:[^#\n]*\b)?dh\b'),
     "DH", AlgoCategory.DH, Severity.HIGH, "cryptography.io DH import"),
    (re.compile(r'dh\.generate_parameters\(|DHParameterNumbers\(|dh\.DHPublicKey\b'),
     "DH", AlgoCategory.DH, Severity.HIGH, "DH usage"),

    # DSA
    (re.compile(r'from\s+cryptography\.hazmat\.primitives\.asymmetric\s+import\s+(?:[^#\n]*\b)?dsa\b'),
     "DSA", AlgoCategory.DSA, Severity.CRITICAL, "cryptography.io DSA import"),
    (re.compile(r'from\s+Crypto(?:dome)?\.PublicKey\s+import\s+(?:[^#\n]*\b)?DSA\b'),
     "DSA", AlgoCategory.DSA, Severity.CRITICAL, "PyCryptodome DSA import"),
    (re.compile(r'dsa\.generate_private_key\(|\bDSAPublicKey\b|\bDSAPrivateKey\b'),
     "DSA", AlgoCategory.DSA, Severity.CRITICAL, "DSA usage"),

    # paramiko / ssh (RSA/DSA/ECDSA keys)
    (re.compile(r'paramiko\.RSAKey|paramiko\.DSSKey|paramiko\.ECDSAKey'),
     "SSH-RSA/DSA/ECDSA", AlgoCategory.RSA, Severity.HIGH, "paramiko quantum-vulnerable key type"),

    # Deprecated
    (re.compile(r'hashlib\.md5\(|MD5\.new\('),
     "MD5", AlgoCategory.HASH, Severity.MEDIUM, "MD5 hash usage"),
    (re.compile(r'hashlib\.sha1\(|SHA\.new\(|SHA1\.new\('),
     "SHA-1", AlgoCategory.HASH, Severity.MEDIUM, "SHA-1 hash usage"),
]

# --- Config file patterns ---
CFG_PATTERNS: list[tuple[re.Pattern, str, AlgoCategory, Severity, str]] = [
    # nginx/Apache: ssl_ciphers '...RSA...' or ssl_ciphers = '...DHE...'
    (re.compile(r'(?i)cipher(?:suite)?s?\s*[=:\s]\s*[^\n]*(?:\bRSA\b|\bECDSA\b|\bECDHE\b|\bDHE\b|\bDH\b)'),
     "TLS cipher with QV algo", AlgoCategory.RSA, Severity.HIGH, "cipher suite config"),
    # nginx/OpenSSL: ssl_protocols TLSv1.0 or ssl_protocol = TLSv1.1
    (re.compile(r'(?i)ssl_protocols?\s*[=:\s]\s*[^\n]*(?:TLSv1\.0|TLSv1\.1|SSLv[23])'),
     "Deprecated TLS", AlgoCategory.RSA, Severity.HIGH, "deprecated TLS version"),
    (re.compile(r'(?i)ssh_host_(?:rsa|dsa|ecdsa)_key'),
     "SSH host key type", AlgoCategory.RSA, Severity.HIGH, "SSH host key config"),
    # key_algorithm = RSA  (with or without "private" prefix)
    (re.compile(r'(?i)(?:private[-_]?)?key[-_]?algorithm\s*[=:\s]\s*(?:rsa|dsa|ecdsa|ec\b)'),
     "Key algorithm config", AlgoCategory.RSA, Severity.HIGH, "key algorithm config"),
    (re.compile(r'(?i)key[-_]?type\s*[=:\s]\s*(?:rsa|dsa|ecdsa)'),
     "Key type config", AlgoCategory.RSA, Severity.HIGH, "key type config"),
]

# --- Binary patterns (ASN.1 OIDs for common QV algorithms embedded in firmware) ---
BINARY_PATTERNS: list[tuple[bytes, str, AlgoCategory, Severity, str]] = [
    # RSA OID: 1.2.840.113549.1.1.1  → 06 09 2a 86 48 86 f7 0d 01 01 01
    (b'\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01',
     "RSA", AlgoCategory.RSA, Severity.CRITICAL, "ASN.1 OID rsaEncryption"),
    # RSA-SHA256: 1.2.840.113549.1.1.11 → 06 09 2a 86 48 86 f7 0d 01 01 0b
    (b'\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x0b',
     "RSA-SHA256", AlgoCategory.RSA, Severity.HIGH, "ASN.1 OID sha256WithRSAEncryption"),
    # EC public key OID: 1.2.840.10045.2.1 → 06 07 2a 86 48 ce 3d 02 01
    (b'\x06\x07\x2a\x86\x48\xce\x3d\x02\x01',
     "EC", AlgoCategory.ECC, Severity.HIGH, "ASN.1 OID id-ecPublicKey"),
    # ECDSA-with-SHA256: 1.2.840.10045.4.3.2 → 06 08 2a 86 48 ce 3d 04 03 02
    (b'\x06\x08\x2a\x86\x48\xce\x3d\x04\x03\x02',
     "ECDSA-SHA256", AlgoCategory.ECC, Severity.HIGH, "ASN.1 OID ecdsa-with-SHA256"),
    # NIST P-256 curve OID: 1.2.840.10045.3.1.7 → 06 08 2a 86 48 ce 3d 03 01 07
    (b'\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07',
     "P-256", AlgoCategory.ECC, Severity.HIGH, "ASN.1 OID P-256 curve"),
    # DSA OID: 1.2.840.10040.4.1 → 06 07 2a 86 48 ce 38 04 01
    (b'\x06\x07\x2a\x86\x48\xce\x38\x04\x01',
     "DSA", AlgoCategory.DSA, Severity.CRITICAL, "ASN.1 OID id-dsa"),
    # DH OID: 1.2.840.10046.2.1 → 06 09 2a 86 48 86 f7 0d 01 03 01
    (b'\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x03\x01',
     "DH", AlgoCategory.DH, Severity.HIGH, "ASN.1 OID dhpublicnumber"),
    # "RSA" ASCII in firmware strings (library names / banners)
    (b'RSA PRIVATE KEY',
     "RSA private key", AlgoCategory.RSA, Severity.CRITICAL, "PEM RSA private key marker"),
    (b'EC PRIVATE KEY',
     "EC private key", AlgoCategory.ECC, Severity.HIGH, "PEM EC private key marker"),
    (b'DSA PRIVATE KEY',
     "DSA private key", AlgoCategory.DSA, Severity.CRITICAL, "PEM DSA private key marker"),
]

# --- File extensions for each scan mode ---
TEXT_SOURCE_EXTS  = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"}
PYTHON_EXTS       = {".py"}
CONFIG_EXTS       = {".conf", ".cfg", ".ini", ".yaml", ".yml", ".toml", ".json", ".xml", ".env"}
CERT_EXTS         = {".pem", ".crt", ".cer", ".key", ".pub", ".der", ".p12", ".pfx", ".csr"}
BINARY_EXTS       = {".bin", ".img", ".hex", ".elf", ".axf", ".out", ".o", ".a", ".so", ".fw"}

# Files with no extension we still try as binary if small enough
MAX_BINARY_SIZE   = 64 * 1024 * 1024  # 64 MB


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class PQCScanner:
    """Scan a directory (or single file) for quantum-vulnerable cryptographic usage."""

    def __init__(
        self,
        path: Path,
        exclude_dirs: Optional[list[str]] = None,
        min_severity: Severity = Severity.INFO,
    ):
        self.root = Path(path)
        self.exclude_dirs = set(exclude_dirs or [".git", "__pycache__", "node_modules", ".venv", "venv"])
        self.min_severity = min_severity
        self._severity_order = [Severity.INFO, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

    def _include_severity(self, s: Severity) -> bool:
        return self._severity_order.index(s) >= self._severity_order.index(self.min_severity)

    # ------------------------------------------------------------------
    # File walkers
    # ------------------------------------------------------------------

    def _walk(self) -> Iterator[Path]:
        if self.root.is_file():
            yield self.root
            return
        for p in self.root.rglob("*"):
            if p.is_file() and not any(ex in p.parts for ex in self.exclude_dirs):
                yield p

    # ------------------------------------------------------------------
    # Scan helpers
    # ------------------------------------------------------------------

    def _scan_text(self, path: Path, content: str, patterns, lineno_offset: int = 1) -> list[Finding]:
        findings = []
        lines = content.splitlines()
        for regex, algo, cat, sev, detail in patterns:
            for i, line in enumerate(lines, lineno_offset):
                m = regex.search(line)
                if m:
                    snippet = line.strip()[:120]
                    findings.append(Finding(
                        file=str(path),
                        line=i,
                        byte_offset=None,
                        algorithm=algo,
                        category=cat,
                        severity=sev,
                        detail=detail,
                        pqc_replacement=PQC_REPLACEMENT[cat],
                        context=snippet,
                    ))
        return findings

    def _scan_binary(self, path: Path, data: bytes) -> list[Finding]:
        findings = []
        for needle, algo, cat, sev, detail in BINARY_PATTERNS:
            offset = 0
            while True:
                idx = data.find(needle, offset)
                if idx == -1:
                    break
                findings.append(Finding(
                    file=str(path),
                    line=None,
                    byte_offset=idx,
                    algorithm=algo,
                    category=cat,
                    severity=sev,
                    detail=detail,
                    pqc_replacement=PQC_REPLACEMENT[cat],
                    context=f"binary @ 0x{idx:x}",
                ))
                offset = idx + len(needle)
        return findings

    def _scan_cert_pem(self, path: Path, content: str) -> list[Finding]:
        """PEM certificates — detect algorithm from header lines."""
        findings = []
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "RSA PRIVATE KEY" in stripped or "RSA PUBLIC KEY" in stripped:
                findings.append(Finding(str(path), i, None, "RSA", AlgoCategory.RSA,
                    Severity.CRITICAL, "PEM RSA key", PQC_REPLACEMENT[AlgoCategory.RSA], stripped))
            elif "EC PRIVATE KEY" in stripped or "EC PARAMETERS" in stripped:
                findings.append(Finding(str(path), i, None, "EC", AlgoCategory.ECC,
                    Severity.HIGH, "PEM EC key", PQC_REPLACEMENT[AlgoCategory.ECC], stripped))
            elif "DSA PRIVATE KEY" in stripped:
                findings.append(Finding(str(path), i, None, "DSA", AlgoCategory.DSA,
                    Severity.CRITICAL, "PEM DSA key", PQC_REPLACEMENT[AlgoCategory.DSA], stripped))
        return findings

    # ------------------------------------------------------------------
    # Main scan
    # ------------------------------------------------------------------

    def scan_file(self, path: Path) -> list[Finding]:
        ext = path.suffix.lower()
        findings: list[Finding] = []

        try:
            if ext in TEXT_SOURCE_EXTS:
                text = path.read_text(errors="replace")
                findings = self._scan_text(path, text, C_PATTERNS)

            elif ext in PYTHON_EXTS:
                text = path.read_text(errors="replace")
                findings = self._scan_text(path, text, PY_PATTERNS)

            elif ext in CONFIG_EXTS:
                text = path.read_text(errors="replace")
                findings = self._scan_text(path, text, CFG_PATTERNS)
                # Also check for inline PEM blocks
                findings += self._scan_cert_pem(path, text)

            elif ext in CERT_EXTS:
                try:
                    text = path.read_text(errors="replace")
                    findings = self._scan_cert_pem(path, text)
                except Exception:
                    pass
                # Also scan raw bytes for DER
                data = path.read_bytes()
                findings += self._scan_binary(path, data)

            elif ext in BINARY_EXTS or ext == "":
                size = path.stat().st_size
                if size <= MAX_BINARY_SIZE:
                    data = path.read_bytes()
                    findings = self._scan_binary(path, data)

        except (PermissionError, OSError):
            pass

        return [f for f in findings if self._include_severity(f.severity)]

    def scan(self) -> "ScanReport":
        all_findings: list[Finding] = []
        files_scanned = 0
        for path in self._walk():
            found = self.scan_file(path)
            all_findings.extend(found)
            files_scanned += 1
        return ScanReport(
            target=str(self.root),
            files_scanned=files_scanned,
            findings=all_findings,
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ScanReport:
    target: str
    files_scanned: int
    findings: list[Finding]
    scanned_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    @property
    def total(self) -> int:
        return len(self.findings)

    @property
    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "target": self.target,
            "scanned_at": self.scanned_at,
            "files_scanned": self.files_scanned,
            "summary": {
                "total_findings": self.total,
                "by_severity": self.by_severity,
            },
            "findings": [f.as_dict() for f in self.findings],
        }, indent=indent)

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            "  PQC IoT Retrofit Scanner — Results",
            "=" * 72,
            f"  Target       : {self.target}",
            f"  Scanned at   : {self.scanned_at}",
            f"  Files scanned: {self.files_scanned}",
            f"  Total findings: {self.total}",
            "",
            "  By severity:",
        ]
        sev_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.INFO]
        for s in sev_order:
            n = self.by_severity.get(s.value, 0)
            lines.append(f"    {s.value:<10} {n}")
        lines.append("")

        if not self.findings:
            lines.append("  ✓ No quantum-vulnerable crypto detected.")
        else:
            # Group by file
            by_file: dict[str, list[Finding]] = {}
            for f in sorted(self.findings, key=lambda x: (x.file, x.line or 0, x.byte_offset or 0)):
                by_file.setdefault(f.file, []).append(f)

            for fname, flist in by_file.items():
                lines.append(f"\n  {fname}")
                lines.append("  " + "-" * (len(fname) + 2))
                for f in flist:
                    loc = f":{f.line}" if f.line else f"+0x{f.byte_offset:x}" if f.byte_offset is not None else ""
                    lines.append(f"    [{f.severity.value:<8}] line{loc}  {f.algorithm}")
                    lines.append(f"             detail : {f.detail}")
                    if f.context:
                        lines.append(f"             context: {f.context[:100]}")
                    lines.append(f"             replace: {f.pqc_replacement}")

        lines += [
            "",
            "=" * 72,
            "  PQC Reference (NIST 2024):",
            "    ML-KEM   (FIPS 203) — Key encapsulation  [replaces RSA-KEM, ECDH, DH]",
            "    ML-DSA   (FIPS 204) — Digital signatures [replaces RSA-sign, ECDSA, DSA]",
            "    SLH-DSA  (FIPS 205) — Stateless hash-based signatures [alternative]",
            "=" * 72,
        ]
        return "\n".join(lines)
