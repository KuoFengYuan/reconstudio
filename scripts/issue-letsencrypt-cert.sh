#!/usr/bin/env bash
#
# Recon Studio — swap the panel's mkcert certificate for a real one.
#
#   sudo scripts/issue-letsencrypt-cert.sh --email you@example.org
#   sudo scripts/issue-letsencrypt-cert.sh --email you@example.org --staging  # rehearsal
#
# Why this works even though the host is on a private IP: certbot proves the
# domain with **DNS-01**, i.e. by writing a TXT record. Let's Encrypt never
# connects to this machine, so `recon.venraas.tw -> 192.168.90.146` being
# unroutable from the internet does not matter at all. What DOES matter is that
# the zone is publicly resolvable, which venraas.tw is.
#
# Afterwards the name-based vhost carries a certificate every browser already
# trusts — no rootCA.pem to install on anyone's laptop. The IP fallback keeps
# mkcert's, because no public CA will sign a bare 192.168.x.x.
#
# Two things to know before running:
#   * Cloud DNS has no per-zone IAM, so the service account gets DNS write access
#     to EVERY zone in its project. If that is too broad, use CNAME delegation:
#     point _acme-challenge.<domain> at a TXT record in a throwaway zone/project
#     and give the key access to that one instead.
#   * The certificate is published in Certificate Transparency logs, so the NAME
#     becomes public knowledge. It resolves to an unroutable address, but the
#     name itself cannot be kept secret.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CREDS="${CREDS:-/etc/letsencrypt/gcp-dns.json}"
GCP_PROJECT="${GCP_PROJECT:-}"
DOMAIN="${RECON_STUDIO_LAN_DOMAIN:-}"
EMAIL="${EMAIL:-}"
STAGING=0
MODE="${MODE:-auto}"          # auto | sa | gcloud-cli
GCLOUD_ZONE="${GCLOUD_ZONE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --email)    EMAIL="$2";       shift 2 ;;
    --domain)   DOMAIN="$2";      shift 2 ;;
    --creds)    CREDS="$2";       shift 2 ;;
    --project)  GCP_PROJECT="$2"; shift 2 ;;
    --staging)  STAGING=1;        shift ;;
    --gcloud-cli) MODE=gcloud-cli; shift ;;
    --sa)         MODE=sa;         shift ;;
    --zone)     GCLOUD_ZONE="$2"; shift 2 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "${EUID}" -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }

if [[ -z "$DOMAIN" ]]; then
  DOMAIN="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?RECON_STUDIO_LAN_DOMAIN=([^[:space:]#]+).*/\2/p' \
            "$REPO_ROOT/local.env" 2>/dev/null | tail -1)"
  : "${DOMAIN:=recon.venraas.tw}"
fi
[[ -n "$EMAIL" ]] || { echo "--email is required (Let's Encrypt sends expiry warnings there)." >&2
                       exit 2; }

# --- the domain has to be resolvable from the public internet ---------------- #
# DNS-01 is validated by Let's Encrypt's own resolvers, not by this machine, so a
# name that only your LAN knows about fails with a confusing NXDOMAIN mid-run.
echo "==> Checking $DOMAIN is visible in public DNS ..."
if ! host -t A "$DOMAIN" 8.8.8.8 >/dev/null 2>&1; then
  echo "    $DOMAIN does not resolve on 8.8.8.8." >&2
  echo "    Add its A record first — see scripts/deploy-nginx-lan.sh's hint." >&2
  exit 1
fi
echo "    OK ($(dig +short "$DOMAIN" @8.8.8.8 | tr '\n' ' '))"

# --- plugin ------------------------------------------------------------------ #
if ! certbot plugins 2>/dev/null | grep -q dns-google; then
  echo "==> Installing python3-certbot-dns-google ..."
  apt-get update -qq
  apt-get install -y -qq python3-certbot-dns-google
fi

# --- how do we prove control of the name? ----------------------------------- #
# `sa`         a service-account key + certbot-dns-google. Best for unattended
#              renewal, but needs someone with setIamPolicy to grant the account
#              access to the zone.
# `gcloud-cli` certbot --manual hooks driving the gcloud CLI as the operator.
#              Needs no new credential at all: a human with roles/editor can
#              already write records, which is all DNS-01 asks for.
if [[ "$MODE" == auto ]]; then
  [[ -s "$CREDS" ]] && MODE=sa || MODE=gcloud-cli
fi

if [[ "$MODE" == sa ]]; then
  if [[ ! -s "$CREDS" ]]; then
    cat >&2 <<EOF

No service-account key at $CREDS, and --sa was requested.

Minting one needs a project Owner: granting the account access to the zone is
setIamPolicy, which roles/editor does not have at the project OR the zone level.
Ask an Owner of the project holding the LIVE zone for ${DOMAIN#*.} for the
narrowest of the two:

  # zone-scoped (preferred — touches only this one zone)
  gcloud dns managed-zones get-iam-policy <ZONE> --project <PROJECT> --format=json > p.json
  # add: {"role":"roles/dns.admin","members":["serviceAccount:<SA>"]} to .bindings
  gcloud dns managed-zones set-iam-policy <ZONE> --policy-file=p.json --project <PROJECT>

  # project-wide (simpler, but reaches every zone in that project)
  gcloud projects add-iam-policy-binding <PROJECT> --member "serviceAccount:<SA>" \\
      --role roles/dns.admin

Meanwhile --gcloud-cli needs none of that. See the header.
EOF
    exit 1
  fi
  chmod 600 "$CREDS"
  challenge=(--dns-google --dns-google-credentials "$CREDS")
else
  # --- find the zone that is REALLY serving this name ----------------------- #
  # Matching on dnsName alone is not enough: this domain has two stale zones with
  # the same dnsName in a different project, and writing the challenge into one
  # of those would publish a TXT record no resolver ever sees. So require the
  # zone's own nameservers to be the ones the parent actually delegates to.
  BASE="${DOMAIN#*.}"
  mapfile -t LIVE_NS < <(dig +short NS "$BASE" | sort)
  [[ ${#LIVE_NS[@]} -gt 0 ]] || { echo "$BASE has no NS records in public DNS." >&2; exit 1; }
  echo "==> $BASE is delegated to: ${LIVE_NS[*]}"

  if [[ -z "$GCLOUD_ZONE" || -z "$GCP_PROJECT" ]]; then
    echo "==> Looking for the matching Cloud DNS zone ..."
    for proj in $(gcloud projects list --format='value(projectId)' 2>/dev/null); do
      while IFS=$'\t' read -r zname zdns zns; do
        [[ "$zdns" == "$BASE." ]] || continue
        printf '%s\n' "${LIVE_NS[@]}" | grep -qxF "$zns" || {
          echo "    skip $proj/$zname — its NS ($zns) is not in the live delegation"; continue; }
        GCP_PROJECT="$proj"; GCLOUD_ZONE="$zname"
        break 2
      done < <(gcloud dns managed-zones list --project "$proj" \
                 --format='value(name,dnsName,nameServers[0])' 2>/dev/null)
    done
  fi
  [[ -n "$GCLOUD_ZONE" && -n "$GCP_PROJECT" ]] || {
    echo "Could not find the live Cloud DNS zone for $BASE — pass --project and --zone." >&2
    exit 1; }
  echo "    using $GCP_PROJECT / $GCLOUD_ZONE"

  # --- whose gcloud credentials do the hooks run with? ---------------------- #
  # certbot runs as root, and root has never logged into gcloud. Point it at the
  # operator's config instead of failing 60 days from now inside a timer.
  OPERATOR="${SUDO_USER:-root}"
  SDK_CONFIG="$(eval echo "~$OPERATOR")/.config/gcloud"
  [[ -d "$SDK_CONFIG" ]] || { echo "No gcloud config at $SDK_CONFIG." >&2; exit 1; }
  acct="$(CLOUDSDK_CONFIG="$SDK_CONFIG" gcloud config get-value account 2>/dev/null)"
  [[ -n "$acct" && "$acct" != "(unset)" ]] || {
    echo "$SDK_CONFIG has no active account — run: gcloud auth login" >&2; exit 1; }
  echo "==> hooks will act as $acct"

  install -d -m 755 /etc/letsencrypt
  cat > /etc/letsencrypt/acme-gcloud.env <<EOF
# Written by scripts/issue-letsencrypt-cert.sh — read by scripts/acme-gcloud-hook.sh.
# certbot renew keeps only the hook command, not this run's environment, so the
# settings have to live somewhere the hook can find them on its own.
# (No backticks in this heredoc: it is unquoted so \$GCP_PROJECT expands, which
#  means a backtick would run as a command and land its output in the file.)
ACME_GCLOUD_PROJECT=$GCP_PROJECT
ACME_GCLOUD_ZONE=$GCLOUD_ZONE
CLOUDSDK_CONFIG=$SDK_CONFIG
EOF
  chmod 644 /etc/letsencrypt/acme-gcloud.env

  HOOK="$REPO_ROOT/scripts/acme-gcloud-hook.sh"
  [[ -x "$HOOK" ]] || { echo "missing or non-executable: $HOOK" >&2; exit 1; }
  challenge=(--manual --preferred-challenges dns
             --manual-auth-hook "$HOOK" --manual-cleanup-hook "$HOOK")
fi

# --- issue -------------------------------------------------------------------- #
# The deploy-hook is stored in the renewal config, so the unattended `certbot
# renew` timer reloads nginx too — without it the panel keeps serving the expired
# certificate until someone notices.
#
# --cert-name is always explicit. Without it certbot matches an existing
# certificate by its DOMAIN SET, so the real run finds the staging lineage,
# calls itself a renewal, and reports "not yet due for renewal; no action
# taken" — leaving you with only the untrusted certificate.
args=(certonly "${challenge[@]}"
      -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL"
      --deploy-hook "systemctl reload nginx")
if [[ "$STAGING" == 1 ]]; then
  args+=(--staging --cert-name "${DOMAIN}-staging")
else
  args+=(--cert-name "$DOMAIN")
fi

echo "==> certbot ${args[*]}"
certbot "${args[@]}"

if [[ "$STAGING" == 1 ]]; then
  echo
  echo "Staging certificate issued — it is NOT trusted, this was only a rehearsal."
  echo "Re-run without --staging to get the real one."
  exit 0
fi

# --- point nginx at it --------------------------------------------------------- #
"$REPO_ROOT/scripts/deploy-nginx-lan.sh" --cert letsencrypt

cat <<EOF

================================================================
https://$DOMAIN/ now serves a publicly-trusted certificate.
Nobody needs rootCA.pem for that URL any more.

  systemctl list-timers certbot     # renewal runs twice a day, 30 days ahead
  certbot renew --dry-run           # rehearse the renewal, including the reload
  openssl s_client -connect $DOMAIN:443 -servername $DOMAIN </dev/null 2>/dev/null \\
    | openssl x509 -noout -issuer   # want: issuer=...Let's Encrypt...

The IP fallback still uses mkcert — no public CA signs a bare IP address.
EOF
