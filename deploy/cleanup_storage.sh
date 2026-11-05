#!/usr/bin/env bash
# One-time storage cleanup for project ai-trading-bot-507921 (Nov 2026) + rules that keep it small afterwards.
#   1. Artifact Registry cleanup policy: keep the 5 newest images (and the tag that is live), delete the rest
#   2. delete image packages no Cloud Run service or job uses (ai-trading-bot, ai-trading-dashboard, ai-trading-scanner)
#   3. delete the old gcr.io repository (images from services deleted on 2026-10-29)
#   4. auto-delete build uploads older than 30 days in the cloudbuild and run-sources buckets
#   5. delete the Oct 17 pre-compact market DB backup
# Deletions cannot be undone. The script first checks that every service and job runs an image it keeps.
set -euo pipefail
PROJECT=ai-trading-bot-507921
REGION=us-central1
REPO=cloud-run-source-deploy
KEEP_PKG=ict-trading-bot

echo "== images in use"
used=$( { gcloud run services list --project "$PROJECT" --region "$REGION" \
            --format="value(spec.template.spec.containers[0].image)";
          gcloud run jobs list --project "$PROJECT" --region "$REGION" \
            --format="value(spec.template.spec.template.spec.containers[0].image)"; } | sort -u)
echo "$used"
if echo "$used" | grep -qv "/$REPO/$KEEP_PKG:"; then
  echo "A service or job uses an image outside $REPO/$KEEP_PKG - stopping, nothing deleted." >&2
  exit 1
fi
live_tags=$(echo "$used" | sed 's/.*://' | sort -u | paste -sd, -)

echo "== 1. cleanup policy on $REPO (keep 5 newest + live tags: $live_tags)"
policy=$(mktemp)
cat > "$policy" <<EOF
[
  {"name": "keep-5-newest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 5}},
  {"name": "keep-live", "action": {"type": "Keep"},
   "condition": {"tagState": "TAGGED", "tagPrefixes": [$(echo "$live_tags" | sed 's/[^,]*/"&"/g')]}},
  {"name": "delete-rest", "action": {"type": "Delete"}, "condition": {"tagState": "ANY"}}
]
EOF
gcloud artifacts repositories set-cleanup-policies "$REPO" --project "$PROJECT" --location "$REGION" \
  --policy "$policy" --no-dry-run
rm -f "$policy"

echo "== 2. unused image packages"
for pkg in ai-trading-bot ai-trading-dashboard ai-trading-scanner; do
  gcloud artifacts packages delete "$pkg" --repository "$REPO" --location "$REGION" --project "$PROJECT" \
    --quiet --async || echo "   $pkg: already gone"
done

echo "== 3. old gcr.io repository"
gcloud artifacts repositories delete gcr.io --location us --project "$PROJECT" --quiet --async \
  || echo "   gcr.io: already gone"

echo "== 4. 30-day lifecycle on build-upload buckets"
lc=$(mktemp)
echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30}}]}' > "$lc"
for b in "${PROJECT}_cloudbuild" "run-sources-${PROJECT}-${REGION}"; do
  gcloud storage buckets update "gs://$b" --lifecycle-file="$lc"
done
rm -f "$lc"

echo "== 5. pre-compact DB backup"
gcloud storage rm "gs://ai-trading-ledger-bucket-464783405434/marketdb/market_data.pre-compact-2026-10-17.db" \
  || echo "   already gone"

echo "Done. Image deletions run in the background; the cleanup policy and bucket lifecycle apply within ~1 day."
