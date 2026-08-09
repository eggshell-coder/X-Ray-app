# Security policy

## Reporting a vulnerability

Please do not open a public issue for security-sensitive information. Contact
the repository owner privately with a description, reproduction steps, and the
potential impact.

Do not upload patient-identifying medical images, credentials, API keys, or
private deployment logs to GitHub issues or pull requests.

## Deployment guidance

- Set `VITE_API_URL` only in the frontend deployment environment.
- Restrict CORS to the deployed frontend origin for production use.
- Keep model files and uploaded medical images in protected storage.
- Rotate any credential that is accidentally committed.
