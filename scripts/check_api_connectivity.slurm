#!/usr/bin/env bash
#SBATCH --job-name=api-connectivity
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/api-connectivity-%j.out
#SBATCH --error=logs/api-connectivity-%j.err
#
# Does a COMPUTE NODE have outbound internet access to the API gateway?
#
# HPC compute nodes are frequently firewalled off from the public internet even
# when login nodes are not. The API benchmark is a pure HTTP client, so if the
# compute node cannot reach the endpoint the job would fail — potentially only
# after a long queue wait. This 10-minute probe answers that up front.
#
# Submit: sbatch scripts/check_api_connectivity.slurm
#
set -uo pipefail   # NOT -e: we want every check to run and report, not abort early

: "${OPENAI_COMPAT_API_KEY:=${DEEPSEEK_API_KEY:-}}"

echo "=== node: $(hostname) ==="
echo

echo "--- 1. DNS resolution ---"
getent hosts api.hcnsec.cn || echo "FAILED: cannot resolve api.hcnsec.cn"
echo

echo "--- 2. proxy environment ---"
env | grep -i "_proxy" || echo "(no proxy variables set)"
echo

echo "--- 3. TCP/TLS reachability (10s timeout) ---"
curl -sS -o /dev/null -w "http_code=%{http_code} connect=%{time_connect}s total=%{time_total}s\n" \
  --max-time 10 https://api.hcnsec.cn/v1/models \
  -H "Authorization: Bearer ${OPENAI_COMPAT_API_KEY}" \
  || echo "FAILED: no HTTP response (blocked, or no route to host)"
echo

echo "--- 4. real chat completion ---"
if [ -z "$OPENAI_COMPAT_API_KEY" ]; then
  echo "SKIPPED: no API key in the job environment."
  echo "         (that itself is a finding — SLURM did not inherit it)"
else
  curl -sS --max-time 60 https://api.hcnsec.cn/v1/chat/completions \
    -H "Authorization: Bearer ${OPENAI_COMPAT_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('FAILED: response was not JSON:', e); raise SystemExit
print('served model:', d.get('model'))
print('content:', (d.get('choices') or [{}])[0].get('message',{}).get('content'))
" || echo "FAILED: request did not complete"
fi
echo

echo "--- 5. via the adapter itself ---"
module purge
module load python/3.12-base
cd /home/atuin/b180dc/b180dc50/german_medical_translator
source .venv/bin/activate
python3 -c "
from medmt_eval.models.factory import create_translator
t = create_translator('openai-compat', model_id='glm-5.2', batch_size=2)
try:
    out = t.translate(['Kein Pleuraerguss.', 'Kein Pneumothorax.'], 'de', 'en')
    print('translations:', out)
    print('served:', t.generation_config['served_models'])
    print('substitution:', t.generation_config['model_substitution'])
except Exception as e:
    print('FAILED:', type(e).__name__, e)
"
echo
echo "=== done ==="
