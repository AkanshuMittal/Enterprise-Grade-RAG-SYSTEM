PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-asia-south1}"                 # Mumbai
AR_REPO="${AR_REPO:-rag-repo}"
SERVICE_NAME="${SERVICE_NAME:-rag-api}"
UI_SERVICE_NAME="${UI_SERVICE_NAME:-rag-ui}"

RUNTIME_SA_NAME="rag-runtime"                       # identity the running containers use
DEPLOYER_SA_NAME="github-deployer"                  # identity GitHub Actions uses
GITHUB_REPO="AkanshuMittal/Enterprise-Grade-RAG-SYSTEM"   # EDIT: e.g. AkanshuMittal/enterprise-agentic-rag
WIF_POOL="github-pool"
WIF_PROVIDER="github-provider"

# Secrets -> Secret Manager (values read from your local .env)
SECRET_KEYS=(RAG_API_KEY GROQ_API_KEY PORTKEY_API_KEY QDRANT_API_KEY JINA_API_KEY LOGFIRE_TOKEN GEMINI_API_KEY)
# Non-secret settings -> plain env vars (values read from your local .env)
PLAIN_KEYS=(PORTKEY_CONFIG_SLUG QDRANT_CLUSTER_ENDPOINT QDRANT_COLLECTION)

API_MEMORY="2Gi"
UI_MEMORY="512Mi"

RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="${DEPLOYER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"

# Read one value from .env (handles quotes and Windows line endings)
env_val() {
  grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}
