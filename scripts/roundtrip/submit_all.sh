#!/usr/bin/env bash
# Submit round-trip runs for every model into ONE shared results directory.
#
#   bash scripts/roundtrip/submit_all.sh            # local models
#   RT_API=1 bash scripts/roundtrip/submit_all.sh   # also the 3 hosted models
#
# All jobs write to results/roundtrip_<timestamp>/ so the curves can be compared
# directly. Override with RT_DIR=results/roundtrip_foo.
set -euo pipefail
cd /home/atuin/b180dc/b180dc50/german_medical_translator
mkdir -p logs

RT_DIR="${RT_DIR:-results/roundtrip_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RT_DIR"
echo "results -> $RT_DIR"

sub() {  # model, script
  local m="$1" s="$2"
  local id
  id=$(RESUME_DIR="$RT_DIR" MODEL="$m" sbatch --parsable --job-name="rt-$m" "$s")
  printf '  %-20s %s  %s\n' "$m" "$id" "$s"
}

for m in identity opus nllb hymt2-1.8b translategemma-4b qwen35-4b hymt2-7b; do
  sub "$m" scripts/roundtrip/mig.slurm
done
for m in hymt2-30b-a3b translategemma-27b qwen35-27b; do
  sub "$m" scripts/roundtrip/large.slurm
done
if [ "${RT_API:-0}" = "1" ]; then
  for m in glm-5.2 DeepSeek-V4-Flash MiniMax-M3; do
    sub "$m" scripts/roundtrip/api.slurm
  done
fi
echo
echo "watch:   squeue -u \$USER | grep rt-"
echo "collect: python3 scripts/roundtrip/collect.py $RT_DIR"
