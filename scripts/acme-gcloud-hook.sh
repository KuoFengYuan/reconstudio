#!/usr/bin/env bash
#
# certbot DNS-01 hook that writes the challenge TXT with the **gcloud CLI**.
#
# Used as both --manual-auth-hook and --manual-cleanup-hook; certbot tells the
# two apart by exporting CERTBOT_AUTH_OUTPUT for the cleanup call.
#
# Why not certbot-dns-google: that plugin only accepts a **service-account key**
# and only looks for the managed zone inside that key's own project. Granting a
# service account access to this zone needs `setIamPolicy`, which `roles/editor`
# does not have — at the project OR the zone level (both were tried and denied).
# A human's own editor role *can* write records, and that is all DNS-01 needs, so
# this hook borrows the operator's gcloud credentials instead of minting new ones.
#
# The trade-off, stated plainly: unattended renewal then depends on one person's
# OAuth refresh token staying valid. If it is ever revoked, renewal fails and
# Let's Encrypt emails the address the cert was registered with. The durable fix
# is a service account — ask a project Owner for a zone-scoped binding:
#
#   gcloud dns managed-zones set-iam-policy venraas-tw-zone --policy-file=p.json \
#       --project venraasitri     # bindings: roles/dns.admin for the SA
#
# Environment:
#   ACME_GCLOUD_PROJECT   project holding the zone      (required)
#   ACME_GCLOUD_ZONE      managed-zone name             (required)
#   CLOUDSDK_CONFIG       gcloud config dir to auth as  (certbot runs as root,
#                                                        so this must point at
#                                                        the operator's ~/.config/gcloud)
set -euo pipefail

# Renewal is unattended: `certbot renew` stores only the hook COMMAND in the
# renewal config, never the environment the first run had. So the settings live
# in a file the hook reads itself, or renewal would fail 60 days later with a
# blank "set ACME_GCLOUD_ZONE" that nobody is watching for.
ENV_FILE="${ACME_GCLOUD_ENV:-/etc/letsencrypt/acme-gcloud.env}"
# shellcheck source=/dev/null
[[ -r "$ENV_FILE" ]] && . "$ENV_FILE"

: "${CERTBOT_DOMAIN:?this script is a certbot hook — certbot sets CERTBOT_DOMAIN}"
: "${ACME_GCLOUD_PROJECT:?set ACME_GCLOUD_PROJECT (or put it in $ENV_FILE)}"
: "${ACME_GCLOUD_ZONE:?set ACME_GCLOUD_ZONE (or put it in $ENV_FILE)}"
export CLOUDSDK_CONFIG

RECORD="_acme-challenge.${CERTBOT_DOMAIN}."

log() { echo "[acme-hook] $*" >&2; }
# The subcommand has to come straight after `record-sets`; gcloud rejects flags
# in front of it ("Invalid choice: 'venraas-tw-zone'").
rs() { local op="$1"; shift
       gcloud dns record-sets "$op" "$@" \
         --project "$ACME_GCLOUD_PROJECT" --zone "$ACME_GCLOUD_ZONE"; }

# --- cleanup pass ------------------------------------------------------------ #
# Dispatch on whether certbot SET the variable, not on whether it is non-empty:
# when the auth hook fails it prints to stderr, so CERTBOT_AUTH_OUTPUT arrives as
# an empty string and an -n test sends the cleanup pass back through auth —
# rewriting the record it was called to remove.
# Always exit 0: a failed cleanup must not fail a successful issuance, and the
# record is harmless (it proves nothing on its own and is replaced next time).
if [[ -n "${CERTBOT_AUTH_OUTPUT+set}" ]]; then
  log "removing $RECORD"
  rs delete "$RECORD" --type TXT >/dev/null 2>&1 \
    && log "removed" || log "nothing to remove (already gone)"
  exit 0
fi

# --- auth pass --------------------------------------------------------------- #
: "${CERTBOT_VALIDATION:?certbot sets CERTBOT_VALIDATION}"
log "writing $RECORD = $CERTBOT_VALIDATION"

# TXT rrdata has to reach Cloud DNS quoted, and `create` fails on an existing
# name — a retry after a half-finished run would otherwise be unrecoverable
# without hand-editing production DNS.
if rs describe "$RECORD" --type TXT >/dev/null 2>&1; then
  rs update "$RECORD" --type TXT --ttl 60 --rrdatas "\"$CERTBOT_VALIDATION\"" >/dev/null
else
  rs create "$RECORD" --type TXT --ttl 60 --rrdatas "\"$CERTBOT_VALIDATION\"" >/dev/null
fi

# --- wait for the authoritative servers -------------------------------------- #
# certbot's manual plugin does NOT wait, so without this the ACME server is asked
# to validate a record that has not landed yet. Ask the zone's own nameservers,
# not a cache: a resolver that just answered NXDOMAIN will keep saying so for the
# whole negative-cache TTL, long after the record exists.
# gcloud joins a repeated field with ';', not tabs — splitting on the wrong one
# yields a single blob that dig then queries as if it were one hostname, so the
# wait always times out even though the record is live.
mapfile -t NS < <(gcloud dns managed-zones describe "$ACME_GCLOUD_ZONE" \
                    --project "$ACME_GCLOUD_PROJECT" --format="value(nameServers)" \
                  | tr ';' '\n' | sed '/^$/d')
[[ ${#NS[@]} -gt 0 ]] || { log "could not read the zone's nameservers"; exit 1; }

for attempt in $(seq 1 30); do
  ok=1
  for ns in "${NS[@]}"; do
    dig +short TXT "$RECORD" "@$ns" 2>/dev/null | grep -qF "$CERTBOT_VALIDATION" || { ok=0; break; }
  done
  if [[ "$ok" == 1 ]]; then
    log "visible on all ${#NS[@]} nameservers after ${attempt}0s"
    sleep 5           # a beat for any ACME-side resolver that just missed it
    exit 0
  fi
  sleep 10
done

log "TXT record never became visible on all nameservers — giving up"
exit 1
