import streamlit as st
from transformers import pipeline
import time

st.set_page_config(
    page_title="DocSnap - AI Document Summarizer",
    page_icon="📄",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
h1 { color: #1C4E79; }
.stButton > button { background-color: #1C4E79; color: white; font-weight: bold; }
.result-box { background: #F0FFF4; border: 1.5px solid #2E7D32;
              border-radius: 8px; padding: 16px; margin-top: 12px; }
.metric-label { font-size: 13px; color: #555; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("DocSnap — AI Document Summarizer")
st.markdown("Intelligent summarization powered by **BART-large-CNN** · Meta AI Research")
st.markdown("*Amity University Online · MBA (Data Science) · Keystone Advisory Group, Chennai · 2025–2026*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_len = st.slider("Max summary length (tokens)", 30, 200, 130, step=10)
    min_len = st.slider("Min summary length (tokens)", 10, 60, 30, step=5)
    st.divider()
    st.markdown("**Model Details**")
    st.caption("facebook/bart-large-cnn")
    st.caption("~406M parameters")
    st.caption("Fine-tuned on CNN/DailyMail")
    st.caption("Max input: 1,024 tokens")
    st.divider()
    st.markdown("**Project Info**")
    st.caption("Amity University Online")
    st.caption("Industry Partner: Keystone Advisory Group")
    st.caption("Academic Year: 2025–2026")

# ── Model Load ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

with st.spinner("⏳ Loading BART-large-CNN model (first load ~2 min)..."):
    summarizer = load_model()
st.success("✓ Model ready — facebook/bart-large-cnn")

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("### 📄 Input Document")
text = st.text_area(
    "Paste your document text below (up to ~700 words for best results):",
    height=280,
    placeholder=(
        "Example: JPMorgan Chase reported third-quarter 2024 net income of $12.9 billion, "
        "or $4.37 per diluted share, compared with net income of $9.7 billion in the prior-year "
        "quarter. Revenue was $43.3 billion, up 6% year-over-year...\n\n"
        "Supports: Financial reports · Clinical notes · News articles · Legal contracts · Research papers"
    )
)

if text:
    wc = len(text.split())
    st.caption(f"📊 Input word count: **{wc}** words")
    if wc > 700:
        st.warning("⚠️ Input exceeds 700 words. Text beyond the 1,024-token limit will be truncated by the model.")

# ── Summarize ─────────────────────────────────────────────────────────────────
if st.button("▶  Summarize Document", type="primary", use_container_width=True):
    if text.strip():
        with st.spinner("Generating summary using BART-large-CNN..."):
            t0 = time.time()
            result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            elapsed = round(time.time() - t0, 2)

        summary = result[0]["summary_text"]
        in_words = len(text.split())
        out_words = len(summary.split())
        reduction = round((1 - out_words / max(in_words, 1)) * 100, 1)

        st.markdown("### 📝 Generated Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Input", f"{in_words} words")
        col2.metric("Summary", f"{out_words} words")
        col3.metric("Reduced by", f"{reduction}%")
        col4.metric("Inference", f"{elapsed}s")

        st.markdown(f"""<div class="result-box">{summary}</div>""", unsafe_allow_html=True)
        st.markdown("")
        st.info(f"ℹ️ Model: `facebook/bart-large-cnn` · Max length: {max_len} · Min length: {min_len} · Beam search")

    else:
        st.warning("Please enter some text to summarize.")

# ── Demo Samples ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🧪 Try a Sample Document")
col_a, col_b, col_c = st.columns(3)

SAMPLES = {
    "💼 Financial": (
        "JPMorgan Chase reported third-quarter 2024 net income of 12.9 billion dollars, or 4.37 "
        "per diluted share, compared with net income of 9.7 billion in the prior-year quarter. "
        "Revenue was 43.3 billion, up 6 percent year-over-year, driven by the Commercial and "
        "Investment Bank and record Asset and Wealth Management results. Return on tangible common "
        "equity was 19 percent. The CET1 capital ratio stood at 15.3 percent, well above regulatory "
        "requirements. CEO Jamie Dimon noted the U.S. economy remains resilient but geopolitical "
        "risks and elevated interest rates continue to present material challenges heading into 2025."
    ),
    "🏥 Healthcare": (
        "The patient, a 58-year-old male with a history of type 2 diabetes and hypertension, "
        "presented to the emergency department with acute chest pain of 3 hours duration. "
        "ECG demonstrated ST-elevation in leads V1 through V4. The patient was taken emergently "
        "for cardiac catheterization, which revealed a 95 percent proximal LAD occlusion. "
        "Successful PCI was performed with placement of a drug-eluting stent. Post-procedure "
        "troponin peaked at 45 ng/mL. The patient was admitted to the CICU for monitoring. "
        "Dual antiplatelet therapy with aspirin and clopidogrel was initiated. "
        "Echocardiography the following day demonstrated preserved left ventricular function "
        "with an ejection fraction of 55 percent. The patient was discharged in stable condition "
        "on day 4 with cardiology follow-up scheduled in 2 weeks."
    ),
    "📰 News": (
        "The Federal Reserve held interest rates steady on Wednesday, maintaining the federal "
        "funds rate in the 5.25 to 5.5 percent range as policymakers assessed the path of "
        "inflation and the strength of the labor market. Fed Chair Jerome Powell said the central "
        "bank remains committed to returning inflation to its 2 percent target but acknowledged "
        "that progress has been uneven. Markets had largely anticipated the decision, pricing in "
        "a high probability of unchanged rates ahead of the meeting. Powell indicated the Fed "
        "could begin cutting rates as early as mid-2025 if inflation continues to moderate, "
        "though he cautioned that any easing would be data-dependent. The decision was unanimous "
        "among the 12 voting members of the Federal Open Market Committee."
    )
}

if col_a.button("💼 Financial Report", use_container_width=True):
    st.session_state["sample_text"] = SAMPLES["💼 Financial"]
    st.rerun()
if col_b.button("🏥 Clinical Note", use_container_width=True):
    st.session_state["sample_text"] = SAMPLES["🏥 Healthcare"]
    st.rerun()
if col_c.button("📰 News Article", use_container_width=True):
    st.session_state["sample_text"] = SAMPLES["📰 News"]
    st.rerun()

if "sample_text" in st.session_state:
    st.text_area("Sample loaded — click Summarize above:", value=st.session_state["sample_text"], height=200)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>DocSnap | Keystone Advisory Group, Chennai | "
    "Amity University Online | MBA (Data Science) | 2025–2026 | "
    "Model: <a href='https://huggingface.co/facebook/bart-large-cnn' target='_blank'>facebook/bart-large-cnn</a>"
    "</small></center>",
    unsafe_allow_html=True
)
