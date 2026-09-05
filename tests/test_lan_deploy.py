"""The LAN-exposure plumbing: run.sh, the nginx template, and its installer.

Nothing here needs nginx or root — these are the mistakes that produce a site
which installs cleanly and then doesn't work, or which takes *other* sites down
with it, so they are worth pinning as text.
"""
import re

from pipeline.config import REPO_ROOT

TEMPLATE = (REPO_ROOT / "infra/nginx/reconstudio.conf.template").read_text()
COMMON = (REPO_ROOT / "infra/nginx/reconstudio-common.conf.template").read_text()
DEPLOY = (REPO_ROOT / "scripts/deploy-nginx-lan.sh").read_text()
ISSUE = (REPO_ROOT / "scripts/issue-letsencrypt-cert.sh").read_text()
RUN_SH = (REPO_ROOT / "run.sh").read_text()
SETUP_SH = (REPO_ROOT / "setup.sh").read_text()
EXAMPLE = (REPO_ROOT / "local.env.example").read_text()

_PLACEHOLDER = re.compile(r"__[A-Z_]+__")


def _default_port(text: str) -> set[str]:
    return set(re.findall(r"\bPORT(?::=|=)(\d+)", text))


def test_every_placeholder_is_substituted_by_the_installer():
    """A placeholder the sed doesn't know about ships an nginx config with a
    literal __FOO__ in it — which fails `nginx -t` and takes the reload with it."""
    substituted = set(re.findall(r"s\|(__[A-Z_]+__)\|", DEPLOY))
    found = set(_PLACEHOLDER.findall(TEMPLATE)) | set(_PLACEHOLDER.findall(COMMON))
    assert found                                             # the test can still fail
    assert found <= substituted


def test_the_installer_refuses_to_ship_an_unsubstituted_config():
    assert "unsubstituted placeholders" in DEPLOY


def test_the_upgrade_map_does_not_collide_with_the_other_site_on_this_box():
    """sirocco.conf defines `map $http_upgrade $connection_upgrade` at http
    context. A second definition of the same variable is `[emerg] duplicate`,
    which stops nginx entirely — every site on the machine, not just this one."""
    assert "$rs_connection_upgrade" in TEMPLATE
    assert not re.search(r"map \$http_upgrade \$connection_upgrade", TEMPLATE)


def test_the_proxy_targets_loopback():
    # The panel is meant to stay unreachable except through this proxy; a
    # proxy_pass to the LAN address would quietly make the bypass work.
    assert "proxy_pass http://127.0.0.1:__PANEL_PORT__;" in COMMON


def test_streaming_survives_an_idle_panel():
    """/ws and the SSE endpoints send nothing while every job is idle, so nginx's
    60s default read timeout would drop the bus and freeze the job list."""
    assert "proxy_read_timeout    1h;" in COMMON
    assert "proxy_set_header Connection        $rs_connection_upgrade;" in COMMON


def test_websocket_upgrade_needs_http_1_1():
    assert "proxy_http_version 1.1;" in COMMON


def test_the_installer_reads_the_port_instead_of_asking():
    """Asking would let the proxy drift from the panel the moment someone answers
    from memory — the exact failure `lan_proxy_check` exists to catch."""
    assert "local.env" in DEPLOY and "PORT=" in DEPLOY


def test_the_default_port_is_the_same_number_everywhere():
    """run.sh said 8074 while setup.sh, local.env.example and the README all said
    8077, so anyone forwarding the documented port hit nothing at all."""
    assert _default_port(RUN_SH) == _default_port(SETUP_SH) == _default_port(EXAMPLE) == {"8077"}


def test_run_sh_does_not_inherit_a_bare_HOST_or_PORT():
    """conda's compiler activation exports HOST=x86_64-conda-linux-gnu; plenty of
    tooling exports PORT. Inheriting either gives a panel that won't bind, or one
    listening somewhere nobody was told about."""
    assert "unset HOST PORT" in RUN_SH
    assert "RECON_STUDIO_HOST" in RUN_SH and "RECON_STUDIO_PORT" in RUN_SH


def test_the_unambiguous_names_are_documented():
    # known_env_vars() derives from this file, so an undocumented knob is also
    # reported as an unknown variable by /doctor.
    assert "RECON_STUDIO_HOST" in EXAMPLE and "RECON_STUDIO_PORT" in EXAMPLE


def test_run_sh_refuses_a_port_something_else_already_holds():
    """Otherwise the survivor on that port is the OLD process serving old code,
    which is indistinguishable from "my change did nothing"."""
    assert "sport = :$PORT" in RUN_SH


def test_the_hostname_is_this_projects_own_not_a_borrowed_one():
    """claude.venraas.tw is the LLM chat on the same box. Reusing its name ties
    the panel's URL to a DNS record that belongs to something else."""
    assert 'DEFAULT_DOMAIN="recon.venraas.tw"' in DEPLOY
    assert "claude.venraas.tw" not in TEMPLATE


def test_the_hostname_is_configurable_from_local_env():
    assert "RECON_STUDIO_LAN_DOMAIN" in DEPLOY
    assert "RECON_STUDIO_LAN_DOMAIN" in EXAMPLE      # so /doctor doesn't call it unknown


def test_an_unresolvable_hostname_still_names_a_url_that_works():
    """A second vhost on its own port answers to the bare IP, and the cert covers
    both — so a missing A record costs the pretty name and nothing else."""
    assert "listen __LAN_IP__:__ALT_PORT__ ssl http2;" in TEMPLATE
    assert "server_name __LAN_IP__ localhost;" in TEMPLATE
    assert 'CERT_NAMES=("$DOMAIN" "$LAN_IP" localhost 127.0.0.1)' in DEPLOY


def test_the_shared_port_vhost_claims_only_its_own_name():
    """:443 is shared with another site by SNI. Claiming that site's names —
    the bare IP and localhost — would make nginx pick a winner silently, and the
    loser just stops answering. Those names live on the fallback port instead."""
    main = TEMPLATE.split("listen __LAN_IP__:__HTTPS_PORT__")[1].split("}")[0]
    assert "server_name __DOMAIN__;" in main
    assert "__LAN_IP__ localhost" not in main


def test_the_installer_refuses_to_shadow_another_sites_name():
    assert "already claims server_name" in DEPLOY
    # Token-exact: `.` is a regex wildcard, so a grep would also fire on
    # reconXvenraas.tw.example.com.
    assert "if (a[i] == d) found = 1" in DEPLOY


def test_a_rejected_config_does_not_reach_a_reload():
    """`nginx -t` is all that stands between a typo here and every site on this
    box going down — so a failure has to leave the previous config live."""
    assert "if ! nginx -t; then" in DEPLOY
    assert 'rm -f "$CONF_LINK"' in DEPLOY


def test_each_vhost_redirects_a_plain_http_hit_to_its_own_port():
    """Sending the fallback's visitors to the shared port hands them the OTHER
    site on that address."""
    assert "error_page 497 =301 https://$host$request_uri;" in TEMPLATE
    assert "error_page 497 =301 https://$host:__ALT_PORT__$request_uri;" in TEMPLATE
    assert "error_page 497" not in COMMON            # it cannot know the port


# --- certificates ------------------------------------------------------------ #
# The two vhosts cannot share one: no public CA signs a bare 192.168.x.x, so the
# name gets a real certificate while the IP fallback keeps mkcert's.
def test_the_two_vhosts_carry_different_certificates():
    assert "ssl_certificate     __MAIN_CERT__;" in TEMPLATE
    assert "ssl_certificate     __SSL_DIR__/cert.pem;" in TEMPLATE
    assert "ssl_certificate" not in COMMON        # shared config cannot decide this


def test_auto_uses_the_real_certificate_as_soon_as_one_exists():
    """Otherwise issuing it would silently change nothing until someone
    remembered to re-point nginx by hand."""
    assert 'LE_DIR="/etc/letsencrypt/live/$DOMAIN"' in DEPLOY
    assert '[[ -s "$LE_DIR/fullchain.pem" ]] && CERT_MODE=letsencrypt || CERT_MODE=mkcert' in DEPLOY


def test_asking_for_a_certificate_that_is_not_there_is_an_error():
    # Falling back to mkcert silently would leave a red padlock the user believes
    # they just fixed.
    assert "No Let's Encrypt certificate at $LE_DIR." in DEPLOY


def test_renewal_reloads_nginx():
    """Without the deploy-hook, unattended renewal writes a new certificate that
    nginx never picks up — the panel serves the expired one until it breaks."""
    assert '--deploy-hook "systemctl reload nginx"' in ISSUE


def test_issuance_checks_the_name_is_public_before_burning_a_rate_limit():
    """DNS-01 is validated by Let's Encrypt's resolvers, not this machine, so a
    LAN-only name fails mid-run with a confusing NXDOMAIN."""
    assert 'host -t A "$DOMAIN" 8.8.8.8' in ISSUE


def test_issuance_does_not_mint_credentials_by_itself():
    """A long-lived key with write access to a zone other services depend on is a
    decision, not a side effect of running a setup script — and here it is not
    even the script's to make: the binding needs an Owner."""
    assert "No service-account key at $CREDS, and --sa was requested." in ISSUE
    assert "set-iam-policy" in ISSUE          # it prints what to ask for
    assert "\ngcloud iam service-accounts keys create" not in ISSUE   # never runs it


# --- the gcloud-CLI challenge path ------------------------------------------- #
# certbot-dns-google only accepts a service-account key and only looks for the
# zone inside that key's own project — and granting a service account access to
# this zone needs setIamPolicy, which roles/editor has at neither the project nor
# the zone level (both were tried, both denied). So the fallback drives the
# gcloud CLI as the operator. Every check below is a bug that actually shipped.
HOOK = (REPO_ROOT / "scripts/acme-gcloud-hook.sh").read_text()


def test_the_cleanup_pass_is_chosen_on_the_variable_being_set_not_truthy():
    """A failed auth hook prints to stderr, so certbot hands the cleanup pass an
    EMPTY CERTBOT_AUTH_OUTPUT. `-n "$VAR"` then sent cleanup back through auth and
    it rewrote the record it was called to delete."""
    assert '[[ -n "${CERTBOT_AUTH_OUTPUT+set}" ]]' in HOOK
    assert '[[ -n "${CERTBOT_AUTH_OUTPUT:-}" ]]' not in HOOK


def test_the_gcloud_subcommand_comes_before_its_flags():
    """`gcloud dns record-sets --zone Z create` is rejected with the zone name
    reported as an invalid choice."""
    assert 'gcloud dns record-sets "$op" "$@"' in HOOK


def test_nameservers_are_split_on_gclouds_separator():
    """gcloud joins a repeated field with ';'. Splitting on tabs yields one blob
    that dig queries as a single hostname, so the propagation wait always timed
    out even with the record live."""
    assert "tr ';' '\\n'" in HOOK
    assert "--format=\"value(nameServers)\" | tr '\\t' '\\n'" not in HOOK


def test_the_hook_waits_for_the_authoritative_servers():
    """certbot's manual plugin does not wait, and asking a cache that just
    answered NXDOMAIN keeps returning NXDOMAIN for the negative-cache TTL."""
    assert 'dig +short TXT "$RECORD" "@$ns"' in HOOK


def test_the_hook_reads_its_settings_from_a_file():
    """`certbot renew` keeps only the hook command, never the environment of the
    run that created it, so renewal 60 days later would have neither."""
    assert 'ENV_FILE="${ACME_GCLOUD_ENV:-/etc/letsencrypt/acme-gcloud.env}"' in HOOK
    assert 'ACME_GCLOUD_PROJECT=$GCP_PROJECT' in ISSUE       # and the issuer writes it


def test_the_env_file_heredoc_has_no_backticks():
    """It is unquoted so $GCP_PROJECT expands — which also means a backtick runs
    as a command and lands its output in the file. That shipped once: the file
    began with the output of `certbot renew`."""
    body = ISSUE.split("cat > /etc/letsencrypt/acme-gcloud.env <<EOF")[1].split("\nEOF")[0]
    assert "`" not in body


def test_staging_and_production_are_separate_lineages():
    """Without an explicit --cert-name, certbot matches an existing certificate
    by its DOMAIN SET: the real run finds the staging lineage, calls itself a
    renewal, and reports "not yet due" — leaving only the untrusted cert."""
    assert '--staging --cert-name "${DOMAIN}-staging"' in ISSUE
    assert '--cert-name "$DOMAIN"' in ISSUE


def test_the_zone_is_matched_on_the_live_delegation_not_just_the_name():
    """This domain has two stale zones with the same dnsName in another project;
    writing the challenge into one publishes a TXT no resolver ever sees."""
    assert 'printf \'%s\\n\' "${LIVE_NS[@]}" | grep -qxF "$zns"' in ISSUE
