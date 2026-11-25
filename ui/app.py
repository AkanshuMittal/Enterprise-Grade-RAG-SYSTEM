import html
import os
import time
import uuid

import requests
import streamlit as st

# Optional deps: present locally (full requirements.txt), absent in the slim UI image.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
except ImportError:
    pass

try:
    import logfire
    _token = os.getenv("LOGFIRE_TOKEN")
    if _token:
        logfire.configure(token=_token)
    else:
        raise RuntimeError("no LOGFIRE_TOKEN")
except Exception:
    class _NoLog:
        def info(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def error(self, *a, **k): pass
        def span(self, *a, **k):
            from contextlib import nullcontext
            return nullcontext()
    logfire = _NoLog()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")   # stays server-side; never sent to the browser

EXAMPLES = [
    ("☸️", "What is Kubernetes and why is it used?"),
    ("⏰", "Explain CronJobs and when to use them"),
    ("🔧", "How do I debug a pod that keeps restarting?"),
    ("🌐", "What are the three main developer tools for managing Databricks jobs programmatically?"),
]

# ---------------------------------------------------------------- page + style
st.set_page_config(page_title="Enterprise Agentic RAG", page_icon="🧠", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 6rem; max-width: 860px;}

/* hero */
.hero {text-align: center; padding: 2.2rem 0 1rem 0;}
.hero h1 {
  font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .4rem 0;
  background: linear-gradient(90deg, #8B7CFF 0%, #4FD1C5 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {color: #9AA3B8; font-size: 1.05rem; margin: 0;}
.pill-row {display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; margin-top: 1rem;}
.pill {
  font-size: .75rem; padding: .22rem .65rem; border-radius: 999px;
  border: 1px solid #263150; background: #121828; color: #B8C0D6;
}

/* example buttons */
div[data-testid="stButton"] > button {
  border-radius: 14px; border: 1px solid #263150; background: #121828;
  padding: .9rem 1rem; text-align: left; transition: all .15s ease;
}
div[data-testid="stButton"] > button:hover {
  border-color: #8B7CFF; background: #171F35; transform: translateY(-1px);
}

/* chat bubbles */
[data-testid="stChatMessage"] {
  background: #121828; border: 1px solid #1D2640; border-radius: 16px;
  padding: 1rem 1.1rem; margin-bottom: .8rem;
}

/* reasoning timeline */
.timeline {border-left: 2px solid #263150; margin: .3rem 0 .2rem .5rem; padding-left: 1rem;}
.tl-step {position: relative; margin: .55rem 0; color: #C9D0E4; font-size: .9rem;}
.tl-step::before {
  content: ""; position: absolute; left: -1.38rem; top: .38rem;
  width: 9px; height: 9px; border-radius: 50%; background: #8B7CFF;
  box-shadow: 0 0 0 3px #121828;
}
.tl-key {color: #4FD1C5; font-weight: 600; margin-right: .35rem;}

/* source cards */
.src-card {
  border: 1px solid #263150; border-radius: 12px; padding: .7rem .9rem;
  background: #0F1424; margin-bottom: .55rem;
}
.src-head {display: flex; align-items: center; gap: .5rem; margin-bottom: .25rem;}
.src-badge {
  font-size: .7rem; font-weight: 700; color: #0B0F1A; background: #4FD1C5;
  border-radius: 6px; padding: .05rem .45rem;
}
.src-text {color: #AEB7CC; font-size: .85rem; line-height: 1.45;}

/* latency chip */
.meta-chip {
  display: inline-block; font-size: .72rem; color: #9AA3B8;
  border: 1px solid #263150; border-radius: 999px; padding: .12rem .6rem; margin-top: .4rem;
}

/* sidebar */
[data-testid="stSidebar"] {background: #0E1322; border-right: 1px solid #1D2640;}
.flow-step {display: flex; gap: .6rem; margin: .55rem 0; font-size: .86rem; color: #C9D0E4;}
.flow-num {
  min-width: 22px; height: 22px; border-radius: 50%; background: #1B2440;
  color: #8B7CFF; font-weight: 700; font-size: .75rem;
  display: flex; align-items: center; justify-content: center;
}
.flow-sub {color: #7E88A3; font-size: .76rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"New session: {st.session_state.session_id}")
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role","content","steps","sources","latency"}
if "pending" not in st.session_state:
    st.session_state.pending = None


# ---------------------------------------------------------------- helpers
def esc(text: str) -> str:
    return html.escape(str(text))


def render_steps(steps):
    items = ""
    for s in steps:
        s = str(s)
        if ":" in s:
            k, v = s.split(":", 1)
            items += f'<div class="tl-step"><span class="tl-key">{esc(k)}</span>{esc(v.strip())}</div>'
        else:
            items += f'<div class="tl-step">{esc(s)}</div>'
    st.markdown(f'<div class="timeline">{items}</div>', unsafe_allow_html=True)


def clean_chunk(text: str) -> str:
    t = str(text).strip()
    if t.upper().startswith("CONTENT:"):
        t = t[8:].strip()
    return t


def render_sources(sources):
    for i, src in enumerate(sources, 1):
        body = clean_chunk(src)
        preview = body[:170].replace("\n", " ") + ("…" if len(body) > 170 else "")
        st.markdown(
            f'<div class="src-card"><div class="src-head"><span class="src-badge">#{i}</span></div>'
            f'<div class="src-text">{esc(preview)}</div></div>',
            unsafe_allow_html=True,
        )
        if len(body) > 170:
            with st.expander(f"Read full chunk {i}"):
                st.write(body)


def render_details(steps, sources, latency):
    if steps:
        with st.expander("🧭 How the agent reasoned", expanded=False):
            render_steps(steps)
    if sources:
        with st.expander(f"📄 Retrieved context ({len(sources)} chunks)", expanded=False):
            render_sources(sources)
    if latency:
        st.markdown(f'<span class="meta-chip">⏱ {latency:.1f}s</span>', unsafe_allow_html=True)


def ask_backend(question: str):
    r = requests.post(
        f"{BACKEND_URL}/query",
        json={"q": question, "thread_id": st.session_state.session_id},
        headers={"x-api-key": RAG_API_KEY},
        timeout=120,
    )
    if r.status_code == 401:
        raise RuntimeError("The UI is not authorised to call the API (check RAG_API_KEY).")
    if r.status_code == 429:
        raise RuntimeError("Too many requests. Please wait a minute and try again.")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧠 Agent OS")
    st.caption("Agentic RAG over enterprise IT documentation")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        """
<div class="flow-step"><div class="flow-num">1</div><div>Guardrails<div class="flow-sub">Prompt Guard 2 + NeMo block unsafe input</div></div></div>
<div class="flow-step"><div class="flow-num">2</div><div>Planner (LangGraph)<div class="flow-sub">Decides: search the docs or answer directly</div></div></div>
<div class="flow-step"><div class="flow-num">3</div><div>Retrieve + Rerank<div class="flow-sub">Qdrant vectors, Jina embeddings and reranker</div></div></div>
<div class="flow-step"><div class="flow-num">4</div><div>Generate<div class="flow-sub">gpt-oss-120b via Portkey, with fallback</div></div></div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(f"Session `{st.session_state.session_id[:8]}`")
    if st.button("🗑️ New chat", width="stretch"):
        logfire.warn(f"Memory wipe: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.pending = None
        st.rerun()

def _pick_example(question: str):
    st.session_state.pending = question


# ---------------------------------------------------------------- empty state
if not st.session_state.messages and not st.session_state.pending:
    st.markdown(
        """
<div class="hero">
  <h1>Enterprise Agentic Assistant</h1>
  <p>Ask about Kubernetes, Intel hardware and enterprise networking. Every answer shows its sources.</p>
  <div class="pill-row">
    <span class="pill">LangGraph</span><span class="pill">Qdrant</span><span class="pill">Jina rerank</span>
    <span class="pill">NeMo Guardrails</span><span class="pill">Portkey</span><span class="pill">RAGAS evaluated</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, (icon, q) in enumerate(EXAMPLES):
        with (c1 if i % 2 == 0 else c2):
            st.button(
                f"{icon}  {q}",
                key=f"ex{i}",
                use_container_width=True,
                on_click=_pick_example,
                args=(q,),
            )

# ---------------------------------------------------------------- history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧠" if m["role"] == "assistant" else "👤"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_details(m.get("steps"), m.get("sources"), m.get("latency"))

# ---------------------------------------------------------------- input
typed = st.chat_input("Ask about your documentation...", max_chars=1000)
prompt = typed or st.session_state.pending
st.session_state.pending = None

if prompt:
    with logfire.span("User chat", user_query=prompt, session_id=st.session_state.session_id):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            t0 = time.time()
            try:
                with st.spinner("Planning, retrieving and writing the answer..."):
                    with logfire.span("Calling RAG backend"):
                        data = ask_backend(prompt)
            except Exception as e:
                logfire.error(f"UI-backend failure: {e}")
                st.error(f"Could not get an answer: {e}")
                st.stop()

            latency = time.time() - t0
            answer = data.get("answer") or "No response."
            steps = data.get("thought_process") or []
            sources = data.get("sources") or []

            box = st.empty()
            shown = ""
            for word in answer.split(" "):
                shown += word + " "
                box.markdown(shown + "▌")
                time.sleep(0.008)
            box.markdown(answer)

            render_details(steps, sources, latency)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "steps": steps, "sources": sources, "latency": latency}
            )
            logfire.info("Chat cycle completed.")
