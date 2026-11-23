set -euo pipefail
source "$(dirname "$0")/config.sh"

TAG="manual-$(date +%Y%m%d-%H%M%S)"
API_IMAGE="${REGISTRY}/${SERVICE_NAME}:${TAG}"
UI_IMAGE="${REGISTRY}/${UI_SERVICE_NAME}:${TAG}"

echo "== Build + push API image via Cloud Build"
gcloud builds submit --config <(printf "steps:\n- name: 'gcr.io/cloud-builders/docker'\n  args: ['build', '-f', 'deploy/Dockerfile', '-t', '%s', '.']\nimages:\n- '%s'\n" "$API_IMAGE" "$API_IMAGE") .

echo "== Build + push UI image via Cloud Build"
gcloud builds submit --tag "$UI_IMAGE" ui

# Build --set-secrets and --set-env-vars from config.sh (only keys that exist)
SECRETS_ARG=""
for key in "${SECRET_KEYS[@]}"; do
  gcloud secrets describe "$key" >/dev/null 2>&1 && SECRETS_ARG+="${key}=${key}:latest,"
done
SECRETS_ARG="${SECRETS_ARG%,}"
ENV_ARG=""
for key in "${PLAIN_KEYS[@]}"; do
  ENV_ARG+="${key}=$(env_val "$key")@"
done
ENV_ARG="${ENV_ARG%@}"

echo "== Deploy API ($SERVICE_NAME)"
# max-instances 1: rate limiter and chat memory live inside one instance.
# allow-unauthenticated: the UI must reach it; the x-api-key check protects it.
gcloud run deploy "$SERVICE_NAME" \
  --image "$API_IMAGE" --region "$REGION" \
  --service-account "$RUNTIME_SA" \
  --memory "$API_MEMORY" --cpu 1 --cpu-boost \
  --min-instances 0 --max-instances 1 \
  --timeout 300 --port 8080 \
  --allow-unauthenticated \
  --set-secrets "$SECRETS_ARG" \
  --set-env-vars "^@^${ENV_ARG}"

API_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')
echo "API URL: $API_URL"

echo "== Deploy UI ($UI_SERVICE_NAME)"
gcloud run deploy "$UI_SERVICE_NAME" \
  --image "$UI_IMAGE" --region "$REGION" \
  --service-account "$RUNTIME_SA" \
  --memory "$UI_MEMORY" --cpu 1 \
  --min-instances 0 --max-instances 1 \
  --timeout 300 --port 8080 \
  --allow-unauthenticated \
  --set-secrets "RAG_API_KEY=RAG_API_KEY:latest" \
  --set-env-vars "BACKEND_URL=${API_URL}"

UI_URL=$(gcloud run services describe "$UI_SERVICE_NAME" --region "$REGION" --format='value(status.url)')
echo
echo "DONE."
echo "  API: $API_URL/health"
echo "  UI : $UI_URL"
