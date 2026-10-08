#!/usr/bin/env bash
# Create/update Cloud Scheduler triggers and pause the old ones that ran the broken jobs.
set -euo pipefail
PROJECT="${PROJECT:-ai-trading-bot-507921}"
REGION="${REGION:-us-central1}"
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"

upsert() {  # name schedule job
  local uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/$3:run"
  local verb=create
  gcloud scheduler jobs describe "$1" --project "$PROJECT" --location "$REGION" >/dev/null 2>&1 && verb=update
  gcloud scheduler jobs "$verb" http "$1" --project "$PROJECT" --location "$REGION" \
    --schedule "$2" --time-zone America/New_York --uri "$uri" --http-method POST \
    --oauth-service-account-email "$SA" --oauth-token-scope https://www.googleapis.com/auth/cloud-platform
}

upsert ict-premarket   "45 8 * * 1-5"       ai-trading-premarket
upsert ict-engine-open "30,45 9 * * 1-5"    ai-trading-engine
upsert ict-engine-day  "*/15 10-15 * * 1-5" ai-trading-engine

for old in ai-trading-postopen-trigger ai-trading-premarket-trigger ai-trading-premarket-schedule \
           ai-trading-postopen-schedule ai-trading-cron; do
  gcloud scheduler jobs pause "$old" --project "$PROJECT" --location "$REGION" 2>/dev/null && echo "paused $old" || true
done
