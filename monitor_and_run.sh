#!/usr/bin/env bash
set -euo pipefail

# --- Config -------------------------------------------------------------------
# Snapshot interval for lightweight samplers (seconds)
INTERVAL="${INTERVAL:-2}"

# Path to the script you want to run
TARGET_SCRIPT="${TARGET_SCRIPT:-./run_ablation_experiments.sh}"

# Directory for logs
LOGROOT="${LOGROOT:-./logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${LOGROOT}/run_${TS}"
mkdir -p "${LOGDIR}"

# Aggressive line-buffering for streaming commands
LB() { stdbuf -oL -eL "$@"; }

# Track background PIDs to clean up
PIDS=()
cleanup() {
  echo "[monitor] cleanup..." | tee -a "${LOGDIR}/monitor.log" || true
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  # Final snapshots
  ( date -Is; dmesg --ctime || true ) &> "${LOGDIR}/dmesg_end.txt" || true
}
trap cleanup EXIT INT TERM

# --- System snapshot (once) ---------------------------------------------------
{
  echo "=== BASELINE ==="
  date -Is
  echo "uname -a:"; uname -a
  echo "lsb_release -a:"; lsb_release -a 2>/dev/null || true
  echo "kernel cmdline:"; cat /proc/cmdline
  echo "CPU:"; lscpu
  echo "Memory:"; grep -E 'Mem(Total|Free|Available|SwapTotal|SwapFree)' /proc/meminfo
  echo "PCI (GPU/NVMe hints):"; lspci -nn | grep -E 'VGA|3D|NVMe|NVIDIA|AMD' || true
  echo "Loaded nvidia modules:"; lsmod | grep -i nvidia || true
} > "${LOGDIR}/baseline.txt"

# --- Suggestions (printed) ----------------------------------------------------
cat > "${LOGDIR}/READ_ME.txt" <<EOF
If you experience a hard lockup/reboot:
- Check ${LOGDIR}/journal_follow.log and dmesg_follow.log for last lines.
- Check mem_cpu_pressure.log to see if PSI spikes (memory/cpu/io).
- Check nvidia_dmon.log (if NVIDIA present) for P-State, util, mem, temps, throttling.
- Check vmstat.log / iostat.log / mpstat.log for stalls or 100% iowait/softirq.
- Check sensors.log for thermal issues.
- Check ablation.stdout.log for the last step/set/video before the crash.
EOF

# --- Monitors -----------------------------------------------------------------
echo "[monitor] logs in: ${LOGDIR}"

# 1) Kernel ring buffer (best for driver/oops)
LB bash -c 'sudo dmesg --follow --human 2>&1' \
  > "${LOGDIR}/dmesg_follow.log" &
PIDS+=("$!")

# 2) Full journal (includes kernel + services; noisy but valuable)
LB bash -c 'sudo journalctl -f -o short-iso 2>&1' \
  > "${LOGDIR}/journal_follow.log" &
PIDS+=("$!")

# 3) vmstat: run queue, memory, swap, io, system, cpu
LB vmstat 1 > "${LOGDIR}/vmstat.log" &
PIDS+=("$!")

# 4) mpstat per-CPU
if command -v mpstat >/dev/null 2>&1; then
  LB mpstat -P ALL 1 > "${LOGDIR}/mpstat.log" &
  PIDS+=("$!")
fi

# 5) iostat extended (per-disk latency/util)
if command -v iostat >/dev/null 2>&1; then
  LB iostat -xz 1 > "${LOGDIR}/iostat.log" &
  PIDS+=("$!")
fi

# 6) NVIDIA telemetry (if available)
if command -v nvidia-smi >/dev/null 2>&1; then
  # dmon: power, util, clocks, mem, temps
  LB nvidia-smi dmon -s pucvmt -d 1 > "${LOGDIR}/nvidia_dmon.log" 2>&1 &
  PIDS+=("$!")
  # regular snapshots of full nvidia-smi
  (
    while :; do
      {
        date -Is
        nvidia-smi
        echo
      } >> "${LOGDIR}/nvidia_smi_snapshots.log"
      sleep "${INTERVAL}"
    done
  ) &
  PIDS+=("$!")
fi

# 7) Temperatures / voltages (if sensors configured)
if command -v sensors >/dev/null 2>&1; then
  (
    while :; do
      { date -Is; sensors; echo; } >> "${LOGDIR}/sensors.log"
      sleep "${INTERVAL}"
    done
  ) &
  PIDS+=("$!")
fi

# 8) Memory/CPU/IO pressure (PSI) + meminfo
(
  while :; do
    {
      date -Is
      echo "--- /proc/pressure/cpu ---";    cat /proc/pressure/cpu
      echo "--- /proc/pressure/memory ---"; cat /proc/pressure/memory
      echo "--- /proc/pressure/io ---";     cat /proc/pressure/io
      echo "--- free -m ---";               free -m
      echo "--- /proc/meminfo (key) ---";   egrep '^(Mem|Swap|Huge|DirectMap)' /proc/meminfo
      echo
    } >> "${LOGDIR}/mem_cpu_pressure.log"
    sleep "${INTERVAL}"
  done
) &
PIDS+=("$!")

# 9) Top processes (CPU/MEM) snapshot
(
  while :; do
    {
      date -Is
      ps -eo pid,ppid,comm,%cpu,%mem,rsz,etime --sort=-%cpu | head -n 40
      echo
    } >> "${LOGDIR}/ps_top.log"
    sleep "${INTERVAL}"
  done
) &
PIDS+=("$!")

# 10) Disk space snapshot (in case logs fill disks)
(
  while :; do
    { date -Is; df -h; echo; } >> "${LOGDIR}/df.log"
    sleep 30
  done
) &
PIDS+=("$!")

# --- Make core dumps possible for user-space (useful if only Python dies) ----
ulimit -c unlimited || true
export PYTHONFAULTHANDLER=1  # helps if Python receives fatal signals

# --- Run your workload --------------------------------------------------------
echo "[monitor] starting: ${TARGET_SCRIPT}"
LB bash -c "${TARGET_SCRIPT}" \
  > >(tee -a "${LOGDIR}/ablation.stdout.log") \
  2> >(tee -a "${LOGDIR}/ablation.stderr.log" >&2)

echo "[monitor] workload finished (exit=$?)"
