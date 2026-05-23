# TLS certificates (optional)

This directory is reserved for TLS key material when you extend `deployment/nginx.conf` with an `listen 443 ssl` server block.

The default Docker Compose stack proxies **HTTP only** on port 80 and does not mount this folder.

Place `fullchain.pem` and `privkey.pem` (or your CA bundle) here and add the corresponding `ssl_certificate` / `ssl_certificate_key` directives to Nginx, then re-enable a volume mount in `docker-compose.yml` if needed.
