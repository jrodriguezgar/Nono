# Code Review Reference

## OWASP Top 10 Checklist

| # | Risk | What to Look For |
|---|------|------------------|
| A01 | Broken Access Control | Missing auth checks, IDOR, path traversal |
| A02 | Cryptographic Failures | Weak hashing, hardcoded secrets, HTTP for sensitive data |
| A03 | Injection | SQL/NoSQL/OS injection, unsanitized user input |
| A04 | Insecure Design | Business logic flaws, missing threat modeling |
| A05 | Security Misconfiguration | Debug mode in prod, default credentials, verbose errors |
| A06 | Vulnerable Components | Outdated dependencies, known CVEs |
| A07 | Authentication Failures | Weak passwords, missing MFA, session fixation |
| A08 | Data Integrity Failures | Deserialization, unsigned updates, CI/CD compromise |
| A09 | Logging & Monitoring | Credentials in logs, missing audit trail |
| A10 | SSRF | Unrestricted URL fetches, internal network access |

## Severity Levels

| Level | Criteria | Action |
|-------|----------|--------|
| **high** | Security vulnerability, data loss, crash | Must fix before merge |
| **medium** | Performance issue, missing validation, poor error handling | Should fix |
| **low** | Style, naming, minor improvement | Nice to have |

## Scoring Guide

| Score | Meaning |
|-------|---------|
| 9-10 | Excellent — production-ready |
| 7-8 | Good — minor improvements needed |
| 5-6 | Acceptable — several issues to address |
| 3-4 | Poor — significant rework needed |
| 1-2 | Critical — fundamental problems |
