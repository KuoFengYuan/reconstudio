#!/usr/bin/env bash
#
# Recon Studio — put the panel on the LAN behind nginx (TLS + basic auth).
#
# The panel itself keeps binding 127.0.0.1. Only nginx listens outward, so the
# URL you hand a colleague survives your VS Code window closing, your ssh
# session dropping, and your laptop going home — which a forwarded port does not.
#
#   sudo scripts/deploy-nginx-lan.sh                 # install / update
#   sudo scripts/deploy-nginx-lan.sh --https-port 9443 --domain recon.example.tw
#   sudo scripts/deploy-nginx-lan.sh --uninstall
#
# Idempotent: re-run it after changing PORT in local.env and it re-points the
# proxy. (`./run.sh --doctor` warns when the two have drifted apart, which is the
# usual reason "the port is up but nobody can connect".)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_NAME="reconstudio"
TEMPLATE="$REPO_ROOT/infra/nginx/reconstudio.conf.template"
SNIPPET_SRC="$REPO_ROOT/infra/nginx/reconstudio-common.conf.template"
CONF_DST="/etc/nginx/sites-available/${SITE_NAME}.conf"
CONF_LINK="/etc/nginx/sites-enabled/${SITE_NAME}.conf"
SNIPPET_DST="/etc/nginx/snippets/${SITE_NAME}-common.conf"
HTPASSWD="/etc/nginx/.htpasswd-${SITE_NAME}"
SSL_DIR="/etc/nginx/ssl/${SITE_NAME}"

# The hostname colleagues type. Project-named on purpose: borrowing another
# service's hostname (claude.venraas.tw is the LLM chat on this same box) makes
# the link unguessable to anyone who wasn't told, and ties this panel's URL to a
# DNS record that belongs to something else.
DEFAULT_DOMAIN="recon.venraas.tw"
HTTPS_PORT="${HTTPS_PORT:-443}"   # name-based vhost, shared with whatever else is there
ALT_PORT="${ALT_PORT:-8443}"      # IP fallback, for when DNS is not cooperating
DOMAIN="${RECON_STUDIO_LAN_DOMAIN:-}"
LAN_IP="${LAN_IP:-}"
PANEL_PORT="${PANEL_PORT:-}"
CERT_MODE="${CERT_MODE:-auto}"   # auto | mkcert | letsencrypt
UNINSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --https-port) HTTPS_PORT="$2"; shift 2 ;;
    --alt-port)   ALT_PORT="$2";   shift 2 ;;
    --cert)       CERT_MODE="$2";  shift 2 ;;
    --domain)     DOMAIN="$2";     shift 2 ;;
    --lan-ip)     LAN_IP="$2";     shift 2 ;;
    --panel-port) PANEL_PORT="$2"; shift 2 ;;
    --uninstall)  UNINSTALL=1;     shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "${EUID}" -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }

if [[ "$UNINSTALL" == 1 ]]; then
  rm -f "$CONF_LINK" "$SNIPPET_DST"
  nginx -t && systemctl reload nginx
  echo "Removed $CONF_LINK (config, cert and htpasswd kept — delete by hand if you"
  echo "really want them gone: $CONF_DST $SSL_DIR $HTPASSWD)"
  exit 0
fi

# --- the panel's port: exactly the value run.sh will bind ------------------- #
# Read from local.env rather than asking, so the proxy cannot drift away from
# the panel just because someone answered a prompt from memory.
if [[ -z "$PANEL_PORT" ]]; then
  PANEL_PORT="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?PORT=([0-9]+).*/\2/p' \
                "$REPO_ROOT/local.env" 2>/dev/null | tail -1)"
  : "${PANEL_PORT:=8077}"   # same default as run.sh
fi

PANEL_HOST="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?HOST=([^[:space:]#]+).*/\2/p' \
              "$REPO_ROOT/local.env" 2>/dev/null | tail -1)"

# Same reason as PORT: read the name from local.env rather than relying on it
# being exported into whatever shell `sudo` happened to build.
if [[ -z "$DOMAIN" ]]; then
  DOMAIN="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?RECON_STUDIO_LAN_DOMAIN=([^[:space:]#]+).*/\2/p' \
            "$REPO_ROOT/local.env" 2>/dev/null | tail -1)"
  : "${DOMAIN:=$DEFAULT_DOMAIN}"
fi
: "${PANEL_HOST:=127.0.0.1}"
if [[ "$PANEL_HOST" != "127.0.0.1" && "$PANEL_HOST" != "localhost" ]]; then
  echo "WARNING: local.env has HOST=$PANEL_HOST."
  echo "         With this proxy in front, the panel should stay on 127.0.0.1 —"
  echo "         otherwise :$PANEL_PORT is still reachable on the LAN with no password."
fi

# --- LAN address ------------------------------------------------------------ #
if [[ -z "$LAN_IP" ]]; then
  LAN_IP="$(ip -4 -o addr show scope global \
            | grep -vE ' (docker|br-|veth|virbr)' \
            | awk '{print $4}' | cut -d/ -f1 | head -1)"
fi
[[ -n "$LAN_IP" ]] || { echo "No global IPv4 address found — pass --lan-ip." >&2; exit 1; }
ip -4 addr show | grep -q "inet $LAN_IP/" || {
  echo "$LAN_IP is not bound to any interface on this host." >&2; exit 1; }

# The cert covers both ways in, so the IP fallback is never the one throwing a
# name-mismatch warning at people.
CERT_NAMES=("$DOMAIN" "$LAN_IP" localhost 127.0.0.1)
if [[ -n "$DOMAIN" ]]; then
  resolved="$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1)"
  if [[ "$resolved" != "$LAN_IP" ]]; then
    echo "WARNING: $DOMAIN resolves to '${resolved:-nothing}', not $LAN_IP."
    echo "         https://$LAN_IP:$HTTPS_PORT/ works right now; the NAME needs an A record:"
    echo "           gcloud dns record-sets create $DOMAIN. --type A --ttl 300 \\"
    echo "               --rrdatas $LAN_IP --zone <the LIVE venraas.tw zone> --project <its project>"
    echo "         (192.168.* in public DNS is deliberate here — same trick as"
    echo "          claude.venraas.tw: resolvable everywhere, routable only on the LAN.)"
  fi
fi

echo "==> panel http://127.0.0.1:$PANEL_PORT  ->  https://${DOMAIN:-$LAN_IP}:$HTTPS_PORT"

# --- packages --------------------------------------------------------------- #
missing=()
for b in nginx mkcert htpasswd; do command -v "$b" >/dev/null || missing+=("$b"); done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "==> Installing: ${missing[*]}"
  apt-get update -qq
  apt-get install -y -qq nginx mkcert libnss3-tools apache2-utils
fi

# --- certificate ------------------------------------------------------------ #
# mkcert keeps its CA under the desktop user's HOME, so issue as that user.
CERT_USER="${SUDO_USER:-$(stat -c %U "$REPO_ROOT")}"
if [[ -s "$SSL_DIR/cert.pem" ]] && \
   openssl x509 -in "$SSL_DIR/cert.pem" -noout -checkend 604800 >/dev/null 2>&1 && \
   { [[ -z "$DOMAIN" ]] || openssl x509 -in "$SSL_DIR/cert.pem" -noout -text \
       | grep -q "DNS:$DOMAIN"; }; then
  echo "==> Certificate already covers ${CERT_NAMES[*]} and is not expiring — keeping it."
else
  echo "==> Issuing mkcert certificate for ${CERT_NAMES[*]} ..."
  sudo -u "$CERT_USER" -H mkcert -install
  tmp="$(sudo -u "$CERT_USER" -H mktemp -d)"
  sudo -u "$CERT_USER" -H bash -c \
    "cd '$tmp' && mkcert -cert-file cert.pem -key-file key.pem ${CERT_NAMES[*]}"
  mkdir -p "$SSL_DIR"
  mv "$tmp/cert.pem" "$SSL_DIR/cert.pem"
  mv "$tmp/key.pem"  "$SSL_DIR/key.pem"
  rmdir "$tmp"
  chown -R root:www-data "$SSL_DIR"; chmod 750 "$SSL_DIR"; chmod 640 "$SSL_DIR"/*.pem
fi
CAROOT="$(sudo -u "$CERT_USER" -H mkcert -CAROOT)"

# --- which certificate the public name gets --------------------------------- #
# mkcert's is fine for the IP fallback but is untrusted everywhere else, so the
# name-based vhost prefers a real one when it exists. `auto` means "use the real
# cert the moment it is issued", which is what makes issue-letsencrypt-cert.sh
# a one-way door instead of a thing you must remember to wire up afterwards.
LE_DIR="/etc/letsencrypt/live/$DOMAIN"
case "$CERT_MODE" in
  auto)        [[ -s "$LE_DIR/fullchain.pem" ]] && CERT_MODE=letsencrypt || CERT_MODE=mkcert ;;
  mkcert|letsencrypt) ;;
  *) echo "--cert must be auto, mkcert or letsencrypt (got: $CERT_MODE)" >&2; exit 2 ;;
esac
if [[ "$CERT_MODE" == letsencrypt ]]; then
  [[ -s "$LE_DIR/fullchain.pem" && -s "$LE_DIR/privkey.pem" ]] || {
    echo "No Let's Encrypt certificate at $LE_DIR." >&2
    echo "Issue one first:  sudo scripts/issue-letsencrypt-cert.sh" >&2
    exit 1; }
  MAIN_CERT="$LE_DIR/fullchain.pem"; MAIN_KEY="$LE_DIR/privkey.pem"
  echo "==> $DOMAIN uses the Let's Encrypt certificate (trusted everywhere)."
else
  MAIN_CERT="$SSL_DIR/cert.pem"; MAIN_KEY="$SSL_DIR/key.pem"
  echo "==> $DOMAIN uses the mkcert certificate (red padlock until rootCA.pem is installed)."
fi

# --- basic auth -------------------------------------------------------------- #
if [[ ! -s "$HTPASSWD" ]]; then
  read -rp "Username for the panel: " AUTH_USER
  htpasswd -c "$HTPASSWD" "$AUTH_USER"
  chown root:www-data "$HTPASSWD"; chmod 640 "$HTPASSWD"
else
  echo "==> $HTPASSWD exists — leaving it. Add a user: sudo htpasswd $HTPASSWD <name>"
fi

# --- don't take someone else's name --------------------------------------- #
# Two server blocks claiming one server_name on one address:port is not an
# error — nginx picks whichever it read first and the other site just stops
# answering. On this box :443 already belongs to another service, so this is a
# real way to break something unrelated while "only adding a site".
for other in /etc/nginx/sites-enabled/*; do
  [[ -e "$other" && "$(readlink -f "$other")" != "$CONF_DST" ]] || continue
  # Token-exact, not a substring match: `.` is a regex wildcard and a grep for
  # recon.venraas.tw would also fire on reconXvenraas.tw.example.com.
  if awk -v d="$DOMAIN" '
       /^[[:space:]]*server_name/ {
         line = $0; sub(/;.*/, "", line)
         sub(/^[[:space:]]*server_name[[:space:]]*/, "", line)
         n = split(line, a, /[[:space:]]+/)
         for (i = 1; i <= n; i++) if (a[i] == d) found = 1
       }
       END { exit !found }' "$other"; then
    echo "$(basename "$other") already claims server_name $DOMAIN — refusing to shadow it." >&2
    echo "Pick another name with --domain, or remove it from that site first." >&2
    exit 1
  fi
done

# --- render + reload --------------------------------------------------------- #
render() {   # render <template> <destination>
  sed -e "s|__LAN_IP__|$LAN_IP|g" \
      -e "s|__HTTPS_PORT__|$HTTPS_PORT|g" \
      -e "s|__ALT_PORT__|$ALT_PORT|g" \
      -e "s|__DOMAIN__|$DOMAIN|g" \
      -e "s|__SNIPPET__|$SNIPPET_DST|g" \
      -e "s|__SSL_DIR__|$SSL_DIR|g" \
      -e "s|__MAIN_CERT__|$MAIN_CERT|g" \
      -e "s|__MAIN_KEY__|$MAIN_KEY|g" \
      -e "s|__HTPASSWD__|$HTPASSWD|g" \
      -e "s|__PANEL_PORT__|$PANEL_PORT|g" \
      "$1" > "$2"
  if grep -q '__[A-Z_]*__' "$2"; then
    echo "Template still has unsubstituted placeholders in $2:" >&2
    grep -o '__[A-Z_]*__' "$2" | sort -u >&2
    rm -f "$2"; exit 1
  fi
}
mkdir -p "$(dirname "$SNIPPET_DST")"
render "$SNIPPET_SRC" "$SNIPPET_DST"
render "$TEMPLATE" "$CONF_DST"
ln -sf "$CONF_DST" "$CONF_LINK"
# `nginx -t` is the only thing between a typo here and every site on this box
# going down on the next reload — so a failure must leave the old config live.
if ! nginx -t; then
  rm -f "$CONF_LINK"
  echo "Config rejected; the site has been disabled again and nginx NOT reloaded." >&2
  exit 1
fi
systemctl reload nginx

URL="https://$DOMAIN$([[ "$HTTPS_PORT" == 443 ]] || echo ":$HTTPS_PORT")/"
ALT="https://$LAN_IP:$ALT_PORT/"
cat <<EOF

================================================================
Done.
  hand out:  $URL          [cert: $CERT_MODE]
  fallback:  $ALT        (works even when DNS doesn't; always mkcert)

  curl -k -o /dev/null -sw '%{http_code}\n' $ALT       # 401 = proxy up, auth on
  curl -k -u '<user>:<pass>' -o /dev/null -sw '%{http_code}\n' $ALT   # 200 = panel up

A 401 you cannot get past usually means the URL landed on a DIFFERENT site
sharing this address — check the realm nginx asks for:
  curl -sk -o /dev/null -D- $URL | grep -i www-authenticate    # want: realm="Recon Studio"

Give colleagues:
  1. the URL above and their username/password
  2. $CAROOT/rootCA.pem  — install it for a green padlock
     (skipping it only costs a browser warning; the link still works)

The panel must be running for it to answer 200:  ./run.sh
Check both halves any time with:                 ./run.sh --doctor
EOF
