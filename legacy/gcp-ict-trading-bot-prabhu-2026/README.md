# Archived: old GCP project `ict-trading-bot-prabhu-2026`

Snapshot of the source behind Cloud Run service `ai-trading-dashboard` in the old project
(deployed 2026-09-26, last source upload in gs://run-sources-ict-trading-bot-prabhu-2026-us-central1).
Archived on 2026-11-19 before that project was deleted. `env.yaml` (secrets) was left out.

Other resources in that project, not archived because they held nothing:
- `grafana-dashboard`: stock `grafana/grafana-oss:latest` image, no storage.
- Cloud SQL `trading-db-instance` (Postgres 15): `trading_logs` database had no tables.
- Memorystore `control-redis`: idle, deleted 2026-11-19.

The live bot runs in project `ai-trading-bot-507921`.
