#!/usr/bin/env bash
# ONE-TIME GCP setup. Run from the project root:  bash deploy/setup_gcp.sh
# Safe to re-run: every step checks whether the thing already exists.
set -euo pipefail
source "$(dirname "$0")/config.sh"

[ -f .env ] || { echo "ERROR: .env not found in $(pwd). Run from project root."; exit 1; }
[ -n "$PROJECT_ID" ] || { echo "ERROR: no project. Run: gcloud config set project <ID>"; exit 1; }
[[ "$GITHUB_REPO" != YOUR_* ]] || { echo "ERROR: edit GITHUB_REPO in deploy/config.sh"; exit 1; }

echo "== Project: $PROJECT_ID  Region: $REGION"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "== 1/6 Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com cloudresourcemanager.googleapis.com

echo "== 2/6 Artifact Registry repo"
gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location "$REGION"

echo "== 3/6 Service accounts"
for sa in "$RUNTIME_SA_NAME" "$DEPLOYER_SA_NAME"; do
  gcloud iam service-accounts describe "${sa}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$sa" --display-name "$sa"
done
# runtime: may read secrets
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$RUNTIME_SA" \
  --role roles/secretmanager.secretAccessor --condition=None >/dev/null
# deployer: may deploy to Cloud Run and push images
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$DEPLOYER_SA" \
    --role "$role" --condition=None >/dev/null
done
# deployer may "act as" the runtime identity when deploying
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member "serviceAccount:$DEPLOYER_SA" --role roles/iam.serviceAccountUser >/dev/null

echo "== 4/6 Secrets (values come from your local .env)"
for key in "${SECRET_KEYS[@]}"; do
  val="$(env_val "$key")"
  if [ -z "$val" ]; then echo "   skip $key (empty in .env)"; continue; fi
  gcloud secrets describe "$key" >/dev/null 2>&1 || gcloud secrets create "$key" --replication-policy=automatic >/dev/null
  printf '%s' "$val" | gcloud secrets versions add "$key" --data-file=- >/dev/null
  echo "   ok   $key"
done

echo "== 5/6 Workload Identity Federation (keyless GitHub -> GCP)"
gcloud iam workload-identity-pools describe "$WIF_POOL" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$WIF_POOL" --location=global --display-name="GitHub pool"
gcloud iam workload-identity-pools providers describe "$WIF_PROVIDER" \
  --workload-identity-pool="$WIF_POOL" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
    --workload-identity-pool="$WIF_POOL" --location=global \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
# only YOUR repo may impersonate the deployer account
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPO}" >/dev/null

WIF_PROVIDER_FULL="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"

echo "== 6/6 Done. Add these 7 GitHub Variables"
echo "   (repo -> Settings -> Secrets and variables -> Actions -> Variables tab)"
cat <<VARS

  GCP_PROJECT_ID    = $PROJECT_ID
  GCP_REGION        = $REGION
  GCP_WIF_PROVIDER  = $WIF_PROVIDER_FULL
  GCP_DEPLOYER_SA   = $DEPLOYER_SA
  AR_REPO           = $AR_REPO
  SERVICE_NAME      = $SERVICE_NAME
  UI_SERVICE_NAME   = $UI_SERVICE_NAME

VARS
