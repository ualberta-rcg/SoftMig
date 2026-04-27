#!/bin/bash
set -euo pipefail

HOST="${1:-rack01-12}"
CFG="/scratch/rahimk/SoftMig/ops/logrotate-softmig.conf"
TRIMMER="/scratch/rahimk/SoftMig/ops/trim_softmig_active_logs.sh"

if [ ! -f "$CFG" ] || [ ! -f "$TRIMMER" ]; then
  echo "missing config or trimmer script under /scratch/rahimk/SoftMig/ops" >&2
  exit 1
fi

echo "[1/5] Installing logrotate config on $HOST"
sudo ssh "$HOST" "install -m 0644 '$CFG' /etc/logrotate.d/softmig"

echo "[2/5] Installing active-log trimmer on $HOST"
sudo ssh "$HOST" "install -m 0755 '$TRIMMER' /usr/local/sbin/softmig-trim-active-logs && ls -l /usr/local/sbin/softmig-trim-active-logs"

echo "[3/5] Installing hourly trigger on $HOST"
sudo ssh "$HOST" "cat > /etc/cron.hourly/softmig-logrotate <<'EOF'
#!/bin/sh
/usr/local/sbin/softmig-trim-active-logs >/dev/null 2>&1 || true
EOF
chmod 0755 /etc/cron.hourly/softmig-logrotate
ls -l /etc/cron.hourly/softmig-logrotate"

echo "[4/5] Validating logrotate config on $HOST"
sudo ssh "$HOST" "logrotate -d /etc/logrotate.d/softmig >/tmp/softmig_logrotate.debug 2>&1 || true; sed -n '1,120p' /tmp/softmig_logrotate.debug"

echo "[5/5] Forcing one rotation + trim run on $HOST"
sudo ssh "$HOST" "logrotate -f /etc/logrotate.d/softmig; /usr/local/sbin/softmig-trim-active-logs; ls -lh /var/log/softmig | sed -n '1,60p'; ls -lh /var/log/softmig-maint.log 2>/dev/null || true"

echo "Done."
