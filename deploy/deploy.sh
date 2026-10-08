#!/usr/bin/env bash
# Build one image and deploy: dashboard service + premarket job + engine job.
# Requires env.yaml (gitignored) - see deploy/env.example.yaml.
set -euo pipefail
PROJECT="${PROJECT:-ai-trading-bot-507921}"
REGION="${REGION:-us-central1}"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/ict-trading-bot:$TAG"
cd "$(dirname "$0")/.."
[ -f env.yaml ] || { echo "env.yaml missing (copy deploy/env.example.yaml)"; exit 1; }

# Refuse to ship code that is not in git (an over-broad ignore rule once dropped src/data/).
missing="$(git ls-files --others --ignored --exclude-standard -- src | grep '\.py$' || true)"
missing="$missing$(git ls-files --others --exclude-standard -- src | grep '\.py$' || true)"
if [ -n "$missing" ]; then echo "Source files not committed:"; echo "$missing"; exit 1; fi

gcloud builds submit --project "$PROJECT" --tag "$IMAGE" .

# Dashboard: one instance (Streamlit sessions), long timeout for backtests over websockets.
gcloud run deploy ai-trading-dashboard --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --env-vars-file env.yaml --memory 2Gi --cpu 1 --timeout 3600 --max-instances 1 \
  --session-affinity --allow-unauthenticated

# Pre-market plan (08:45 ET).
gcloud run jobs deploy ai-trading-premarket --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --command python --args=-m,src.app,premarket --env-vars-file env.yaml \
  --memory 1Gi --task-timeout 15m --max-retries 1

# Engine: each run loops 14 minutes (30 s exit checks, scan per 5-minute bar); scheduled every 15 min.
gcloud run jobs deploy ai-trading-engine --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --command python --args=-m,src.app,run,--minutes,14 --env-vars-file env.yaml \
  --memory 1Gi --task-timeout 16m --max-retries 0

echo "Deployed $IMAGE. Run deploy/scheduler.sh once to (re)create the schedules."
