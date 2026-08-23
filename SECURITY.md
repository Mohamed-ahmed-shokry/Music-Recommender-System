# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.4.x   | ✅        |
| < 0.4   | ❌        |

## Reporting a vulnerability

Please do not open a public issue for security reports.

Email the maintainer at `mohamed.shokry.m3as@gmail.com` with:

- Affected version/commit
- Reproduction steps or proof-of-concept
- Impact assessment

We aim to acknowledge within 72 hours and provide a fix or mitigation plan within 14 days. Coordinated disclosure is appreciated; please allow time to patch before public release.

## Hardening in this project

- Request bodies are limited to 64 KiB and validated before parsing.
- Artifact loads verify version, mappings, matrix dimensions, content alignment, and fingerprints.
- Model, mapping, and artifact writes use atomic replacement with fsync.
- CI pins GitHub Actions to immutable commits and disables persisted credentials.
- Container images run as non-root with read-only root filesystems.
