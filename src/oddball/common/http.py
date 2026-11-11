"""Shared HTTP settings for external services.

Queries to public services must be anonymous: no names, e-mail addresses, account handles or
project identifiers in headers, URLs or payloads (``CLAUDE.md``). Every module that talks to an
outside service takes its User-Agent from here.
"""

ANON_USER_AGENT = "astro-research-client/1.0"
