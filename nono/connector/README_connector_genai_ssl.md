# SSL Configuration Guide

This module provides three options for configuring SSL certificate verification when connecting to generative AI services.

## Available Options

### Option 1: INSECURE Mode (Default)

**⚠️ WARNING: For development/testing only**

This mode completely disables SSL verification. Useful for rapid development but **MUST NOT BE USED IN PRODUCTION**.

```python
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Already configured by default, but you can call it explicitly:
configure_ssl_verification(SSLVerificationMode.INSECURE)
```

### Option 2: CERTIFI Mode (Recommended for Production)

**✓ RECOMMENDED for production environments**

This mode uses the `certifi` package which provides updated Mozilla root certificates.

```python
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Configure before using any AI service
configure_ssl_verification(SSLVerificationMode.CERTIFI)
```

If `certifi` is not installed, the module will attempt to install it automatically.

### Option 3: CUSTOM Mode (Corporate Certificates)

**✓ For corporate environments with custom certificates**

This mode allows you to specify a custom certificate file (e.g., your company's root certificate).

```python
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Specify path to corporate certificate
configure_ssl_verification(
    SSLVerificationMode.CUSTOM,
    custom_cert_path=r'C:\certificates\atresmedia-root-ca.crt'
)
```

## Usage in Python Module (main.py)

To change SSL configuration in the main file, add these lines **BEFORE** importing or using any AI service:

```python
# At the beginning of the file, after basic imports
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Option 1: Development (insecure)
configure_ssl_verification(SSLVerificationMode.INSECURE)

# Option 2: Production with certifi (recommended)
# configure_ssl_verification(SSLVerificationMode.CERTIFI)

# Option 3: Corporate certificate
# configure_ssl_verification(
#     SSLVerificationMode.CUSTOM,
#     custom_cert_path=r'C:\certificates\atresmedia-root-ca.crt'
# )
```

## Environment Variables

Each mode configures different environment variables:

### INSECURE Mode:

- `PYTHONHTTPSVERIFY=0`
- `CURL_CA_BUNDLE=''`
- `REQUESTS_CA_BUNDLE=''`

### CERTIFI Mode:

- `REQUESTS_CA_BUNDLE=<path to certifi>`
- `SSL_CERT_FILE=<path to certifi>`
- `CURL_CA_BUNDLE=<path to certifi>`

### CUSTOM Mode:

- `REQUESTS_CA_BUNDLE=<custom path>`
- `SSL_CERT_FILE=<custom path>`
- `CURL_CA_BUNDLE=<custom path>`

## Troubleshooting

### Error: `SSL_ERROR_SSL: CERTIFICATE_VERIFY_FAILED`

This error occurs when Python cannot verify the server's SSL certificate.

**Solutions:**

1. **Rapid development:** Use INSECURE mode (already configured by default)
2. **Production:** Use CERTIFI mode
3. **Behind corporate proxy:** Use CUSTOM mode with your company's certificate

### How to obtain the corporate certificate?

On Windows, you can export the root certificate:

1. Open Certificate Manager: `certmgr.msc`
2. Navigate to: Trusted Root Certification Authorities > Certificates
3. Find your company's certificate (e.g., "Atresmedia Root CA")
4. Right-click > All Tasks > Export
5. Select "Base-64 encoded X.509 (.CER)" format
6. Save the file and use its path in CUSTOM mode

### Verify active mode

The module prints a message to the console when configured:

- `⚠️ SSL verification DISABLED` → INSECURE Mode
- `✓ SSL verification enabled using certifi` → CERTIFI Mode
- `✓ SSL verification enabled using custom certificate` → CUSTOM Mode

## Migration to Production

**IMPORTANT:** Before moving to production, change mode from INSECURE to CERTIFI:

```python
# In main_guardianZ.py, change from:
configure_ssl_verification(SSLVerificationMode.INSECURE)

# To:
configure_ssl_verification(SSLVerificationMode.CERTIFI)
```

This ensures SSL connections are secure and verified.

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `certifi` | >= 2023.0.0 | SSL certificates for secure connections |
| `requests` | >= 2.28.0 | HTTP library with SSL support |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).
