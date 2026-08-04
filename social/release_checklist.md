# Public release checklist

- Replace `[repository link]` in `social/linkedin_post.md`.
- Select software and documentation licences; `NOTICE.md` currently reserves rights.
- Add DOI or archival link to `CITATION.cff` when available.
- Run `make test`, `make quick` and `python scripts/release_audit.py`.
- Confirm `WHITEPAPER.md` and every image render correctly on GitHub.
- Use `social/linkedin_banner.png` as the link preview or post image.
- Create a GitHub release containing the source archive only; no PDF attachments are required.
