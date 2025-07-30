# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do NOT report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in PQC IoT Retrofit Scanner, please report it to:

📧 **security@terragon.ai**

### What to Include

Please include the following information:

- **Type of vulnerability** (e.g., buffer overflow, injection, etc.)
- **Location** of the vulnerability (file, function, line number)
- **Step-by-step reproduction** instructions
- **Proof-of-concept or exploit code** (if available)
- **Impact assessment** (how the vulnerability could be exploited)
- **Suggested fix** (if you have one)

### Response Timeline

- **Initial Response**: Within 24 hours
- **Triage**: Within 3 business days
- **Resolution**: Based on severity (Critical: 7 days, High: 14 days, Medium: 30 days)

### Disclosure Policy

We follow **responsible disclosure**:

1. **Report received** → Acknowledgment sent
2. **Vulnerability confirmed** → Investigation begins
3. **Fix developed** → Security patch prepared
4. **Patch released** → Public disclosure (90 days maximum)
5. **Credit given** → Reporter acknowledged (if desired)

## Security Considerations

### Defensive Security Focus

This tool is designed for **defensive security purposes only**:

✅ **Intended Use**:
- Vulnerability assessment of owned devices
- Post-quantum cryptography migration planning
- Compliance auditing
- Security research with proper authorization

❌ **Not Intended For**:
- Unauthorized access to systems
- Malware development
- Exploitation of vulnerabilities
- Any malicious activities

### Firmware Handling

When processing firmware images:

- **Only analyze firmware you own** or have explicit permission to examine
- **Isolate analysis environment** from production networks
- **Sanitize inputs** to prevent malicious firmware from affecting the scanner
- **Handle keys securely** - never log or store cryptographic keys

### PQC Implementation Security

Our post-quantum cryptography implementations:

- **Follow NIST standards** for standardized algorithms
- **Include side-channel protections** where feasible
- **Use constant-time implementations** to prevent timing attacks
- **Validate all inputs** to prevent injection attacks

### Known Limitations

Current security limitations:

1. **Analysis Environment**: Firmware analysis requires privileged system access
2. **Binary Parsing**: Malformed firmware could potentially cause crashes
3. **PQC Implementations**: Early-stage implementations may have undiscovered vulnerabilities
4. **Dependencies**: Security depends on third-party analysis tools

## Security Best Practices

### For Users

- **Update regularly** to get latest security fixes
- **Validate firmware sources** before analysis
- **Use isolated environments** for untrusted firmware
- **Review generated patches** before deployment
- **Test thoroughly** in non-production environments

### For Contributors

- **Follow secure coding practices**
- **Validate all inputs** from external sources
- **Use safe string handling** functions
- **Implement proper error handling**
- **Add security-focused tests**
- **Document security implications** of changes

## Vulnerability Disclosure Examples

### Accepted Report Types

- **Buffer overflows** in firmware parsing
- **Injection vulnerabilities** in CLI parameters
- **Timing attacks** against PQC implementations
- **Privilege escalation** vulnerabilities
- **Cryptographic implementation flaws**

### Out of Scope

- **Social engineering attacks**
- **Physical device attacks** (this is a software tool)
- **DoS through resource exhaustion** (expected for large firmware)
- **Issues in third-party dependencies** (report to upstream)

## Contact Information

- **Security Issues**: security@terragon.ai
- **General Security Questions**: Open a GitHub Discussion
- **Bug Reports**: GitHub Issues (for non-security bugs only)

## Security Hall of Fame

We recognize security researchers who help improve our security:

<!-- Researchers who report valid vulnerabilities will be listed here -->

*No vulnerabilities reported yet.*

---

**Thank you for helping keep PQC IoT Retrofit Scanner secure!**