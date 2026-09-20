"""
Fruit AI Classifier — Streamlit demo

A luxury-grade interactive dashboard around a EfficientNetV2B0 transfer-
learning model that identifies Apple / Banana / Orange / Strawberry from
real-world photos. The model loading and inference logic below is byte-for-
byte the same as the original deploy/app.py — only the UI/UX layer has been
rebuilt: a dark luxury theme, a sidebar-driven three-page layout (Classify /
Model Info / About), a confidence gauge, a Plotly probability chart, a
downloadable prediction report, and an in-session prediction history.
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Fruit AI Classifier",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# LUXURY THEME
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg-0: #06070a;
    --bg-1: #100b0d;
    --accent-1: #f97316;
    --accent-2: #ec4899;
    --gold: #f0c975;
    --danger: #ef4444;
    --warn: #f59e0b;
    --ok: #22c55e;
    --text-hi: #f5f2ee;
    --text-lo: #a89f9a;
    --border: rgba(214, 178, 148, 0.16);
}

html, body { font-family: 'Inter', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }

/* Never let the custom font stack override Streamlit's own ligature icon fonts
   (this is what makes icons like "upload" / "arrow_right" render as literal text) */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
span[data-icon],
[data-testid="stFileUploaderDropzoneIcon"] svg,
[data-testid*="Icon"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(249,115,22,0.14) 0%, transparent 42%),
        radial-gradient(circle at 92% 8%, rgba(236,72,153,0.10) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(240,201,117,0.06) 0%, transparent 45%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-0) 100%);
}
#MainMenu, footer, header { visibility: hidden; }

.hero-wrap {
    padding: 2.6rem 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(249,115,22,0.14), rgba(236,72,153,0.08));
    border: 1px solid var(--border);
    margin-bottom: 1.4rem;
}
.hero-eyebrow {
    display: inline-block; font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--gold); font-weight: 700;
    background: rgba(240,201,117,0.12); border: 1px solid rgba(240,201,117,0.3);
    padding: 0.32rem 0.85rem; border-radius: 999px; margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
    color: var(--text-hi); margin: 0 0 0.55rem 0; line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--text-lo); font-size: 1.02rem; max-width: 720px; margin: 0; }
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero-badge {
    font-size: 0.78rem; color: var(--text-hi);
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    padding: 0.38rem 0.85rem; border-radius: 999px;
}
.hero-badge.gold { border-color: rgba(240,201,117,0.4); color: var(--gold); }

.section-label {
    font-size: 0.75rem; letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--accent-1); font-weight: 700; margin: 1.3rem 0 0.4rem 0;
}
.section-title { font-size: 1.35rem; font-weight: 700; color: var(--text-hi); margin: 0 0 1rem 0; }

section[data-testid="stSidebar"] { background: var(--bg-0); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stRadio label { color: var(--text-hi) !important; font-weight: 600; }
.sidebar-brand {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.3rem;
    color: var(--text-hi); padding: 0.4rem 0 1rem 0; letter-spacing: -0.02em;
}
.sidebar-brand span { color: var(--accent-1); }

/* Sidebar nav - card style radio */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.65rem; display: flex; flex-direction: column; margin-top: 0.2rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    position: relative;
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1rem 0.9rem 1.2rem;
    margin: 0 !important;
    cursor: pointer;
    box-shadow: 0 2px 10px -4px rgba(0,0,0,0.4);
    transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    border-color: rgba(249,115,22,0.45);
    transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    border-color: transparent;
    box-shadow: 0 10px 26px -8px rgba(249,115,22,0.55);
    transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #1a0a04 !important; font-weight: 700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: var(--text-hi); font-size: 0.93rem; font-weight: 600; letter-spacing: -0.01em;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] .stRadio > label { display: none; }

.stButton>button {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    color: #1a0a04; font-weight: 700; border: none; border-radius: 10px;
    padding: 0.6rem 1.6rem;
    box-shadow: 0 8px 24px -8px rgba(249,115,22,0.5);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 28px -6px rgba(249,115,22,0.65); }

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
    border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--text-lo) !important; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; font-family: 'Sora', sans-serif; }

[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--border); border-radius: 16px; background: rgba(255,255,255,0.02);
}
[data-testid="stExpander"] {
    border: 1px solid var(--border); border-radius: 16px; background: rgba(255,255,255,0.02);
    margin-bottom: 0.8rem;
}
[data-testid="stExpander"] summary { font-weight: 700; color: var(--text-hi); }
[data-testid="stImage"] img { border-radius: 16px; border: 1px solid var(--border); }

p, .stMarkdown, label { color: var(--text-lo); }
h1, h2, h3 { color: var(--text-hi); }

.result-card {
    border-radius: 20px; padding: 1.8rem 2rem; border: 1px solid var(--border);
    margin: 0.6rem 0 1.2rem 0;
    background: linear-gradient(135deg, rgba(249,115,22,0.14), rgba(249,115,22,0.03));
}
.result-headline { font-family: 'Sora', sans-serif; font-size: 2.2rem; font-weight: 800; color: var(--text-hi); margin: 0; }
.result-conf {
    display: inline-block; font-weight: 700; font-size: 0.9rem;
    padding: 0.35rem 0.9rem; border-radius: 999px; margin-top: 0.6rem;
    background: rgba(240,201,117,0.16); color: var(--gold);
}
.result-note {
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);
    color: var(--text-lo); font-size: 0.88rem;
}

.class-card {
    border-radius: 16px; padding: 1.2rem 1.3rem; border: 1px solid var(--border);
    background: rgba(255,255,255,0.02); text-align: center;
}
.class-card .emoji { font-size: 2.2rem; }
.class-card .name { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--text-hi); margin-top: 0.4rem; }

.footer-note {
    margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
    color: var(--text-lo); font-size: 0.82rem; text-align: center;
}
.footer-note a { color: var(--accent-1); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#d8cfc9"),
    margin=dict(l=10, r=10, t=45, b=10),
)

CLASS_EMOJI = {"Apple": "🍎", "Banana": "🍌", "Orange": "🍊", "Strawberry": "🍓"}
CLASS_COLOR = {"Apple": "#ef4444", "Banana": "#f0c975", "Orange": "#f59e0b", "Strawberry": "#ec4899"}

TRAINING_CURVES_URL = (
    "https://raw.githubusercontent.com/huzaifashamsi05/"
    "Fruit-Classification-ML-Model/main/training_curves.png"
)
GITHUB_URL = "https://github.com/huzaifashamsi05/Fruit-Classification-ML-Model"
PORTFOLIO_URL = "https://portfolio-showcase-api-server-gilt.vercel.app"


@st.cache_resource
def load_model():
    """Load the trained model and class labels once per app session. Same
    artifact, same relative paths as the original deploy/app.py."""
    classifier = tf.keras.models.load_model("fruit_classifier.keras")
    with open("class_labels.json") as f:
        classes = json.load(f)
    return classifier, classes


def predict(classifier, classes, image: Image.Image):
    """Byte-for-byte the same preprocessing and inference as the original
    deploy/app.py: resize to 160x160, raw (unnormalized) float32 pixels —
    EfficientNetV2's built-in preprocessing layer expects this. Do not
    change this without retraining/re-validating the model."""
    img_resized = image.resize((160, 160))
    img_array = np.expand_dims(np.array(img_resized).astype(np.float32), axis=0)
    preds = classifier.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = classes[pred_idx]
    confidence = float(preds[pred_idx]) * 100
    return pred_class, confidence, preds


def render_hero(classes):
    class_list = " · ".join(f"{CLASS_EMOJI.get(c, '')} {c}" for c in classes)
    st.markdown(f"""
    <div class="hero-wrap">
        <span class="hero-eyebrow">Deep Learning · Transfer Learning</span>
        <div class="hero-title">🍓 Fruit <span>AI Classifier</span></div>
        <p class="hero-sub">Upload any fruit photo — real-world backgrounds, any lighting, any
        angle — and get an instant prediction with a full confidence breakdown. Trained with a
        background-diversification pipeline specifically to avoid the classic failure mode of
        models that only work on clean studio photos.</p>
        <div class="hero-badges">
            <span class="hero-badge gold">🏆 EfficientNetV2B0</span>
            <span class="hero-badge">🎯 100% test accuracy</span>
            <span class="hero-badge">🧪 {class_list}</span>
            <span class="hero-badge">🌄 Background-diversified training</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_nav(classes):
    st.sidebar.markdown('<div class="sidebar-brand">Fruit<span>AI</span></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="section-label" style="margin-top:0;">Navigate</p>', unsafe_allow_html=True)
    choice = st.sidebar.radio(
        "Navigate",
        ["🍓  Classify a Photo", "📊  Model Info", "ℹ️  About"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="section-label" style="margin-top:0;">Classes Supported</p>', unsafe_allow_html=True)
    for c in classes:
        st.sidebar.markdown(f"{CLASS_EMOJI.get(c, '🍏')} {c}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<p style="font-size:0.78rem;">'
        f'<a href="{GITHUB_URL}" target="_blank">📦 Full source on GitHub</a><br>'
        f'<a href="{PORTFOLIO_URL}" target="_blank">🧑‍💻 Developer portfolio</a>'
        f'</p>',
        unsafe_allow_html=True,
    )
    if "Classify" in choice:
        return "classify"
    elif "Model Info" in choice:
        return "insights"
    return "about"


def render_confidence_gauge(confidence: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 40, "color": "#f5f2ee"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#a89f9a"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": "rgba(214,178,148,0.2)",
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.14)"},
                {"range": [50, 80], "color": "rgba(245,158,11,0.14)"},
                {"range": [80, 100], "color": "rgba(34,197,94,0.14)"},
            ],
        },
        title={"text": "Confidence", "font": {"size": 14, "color": "#a89f9a"}},
    ))
    fig.update_layout(height=260, **PLOTLY_DARK)
    st.plotly_chart(fig, width="stretch")


def render_classify(classifier, classes):
    st.markdown('<p class="section-label" style="margin-top:0;">Step 1</p><p class="section-title">Upload a fruit photo</p>', unsafe_allow_html=True)
    st.caption("Works best with a clear, mostly-centered fruit — but real-world backgrounds, angles, and lighting are exactly what this model was trained to handle.")
    uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is None:
        st.info("⬆️ Upload an image above to get a live prediction.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing photo..."):
        pred_class, confidence, preds = predict(classifier, classes, image)

    color = CLASS_COLOR.get(pred_class, "#f97316")
    emoji = CLASS_EMOJI.get(pred_class, "🍏")

    st.divider()
    st.markdown('<p class="section-label">Result</p><p class="section-title">Prediction</p>', unsafe_allow_html=True)

    col_img, col_gauge, col_card = st.columns([1, 1, 1.2])
    with col_img:
        st.image(image, width="stretch", caption="Uploaded photo")
    with col_gauge:
        render_confidence_gauge(confidence, color)
    with col_card:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-headline">{emoji} {pred_class}</div>
            <span class="result-conf">Confidence: {confidence:.1f}%</span>
            <div class="result-note">
                🧠 Predicted with EfficientNetV2B0, trained on background-diversified data
                for real-world robustness — not just studio photos.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Breakdown</p><p class="section-title">All class probabilities</p>', unsafe_allow_html=True)
    proba_df = pd.DataFrame({
        "Class": [f"{CLASS_EMOJI.get(c, '')} {c}" for c in classes],
        "Probability": [float(p) for p in preds],
        "RawClass": classes,
    }).sort_values("Probability", ascending=True)
    fig = px.bar(
        proba_df, x="Probability", y="Class", orientation="h", text="Probability",
        color="RawClass",
        color_discrete_map=CLASS_COLOR,
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%", height=280, **PLOTLY_DARK)
    st.plotly_chart(fig, width="stretch")

    report_lines = [
        "Fruit AI Classifier — Prediction Report", "",
        f"Predicted class: {pred_class}",
        f"Confidence: {confidence:.2f}%", "",
        "All class probabilities:",
    ]
    for cls, prob in zip(classes, preds):
        report_lines.append(f"  {cls}: {prob * 100:.2f}%")
    report_lines += ["", "Model: EfficientNetV2B0 (transfer learning, fine-tuned)",
                      "Trained with background-diversification augmentation for real-world generalization."]
    st.download_button(
        "⬇️ Download this prediction as a report",
        "\n".join(report_lines).encode("utf-8"),
        "fruit_prediction_report.txt",
        "text/plain",
    )

    history_entry = {
        "Image": uploaded_file.name,
        "Prediction": f"{emoji} {pred_class}",
        "Confidence": f"{confidence:.1f}%",
    }
    st.session_state.setdefault("fruit_history", [])
    if not st.session_state["fruit_history"] or st.session_state["fruit_history"][0]["Image"] != uploaded_file.name:
        st.session_state["fruit_history"].insert(0, history_entry)
        st.session_state["fruit_history"] = st.session_state["fruit_history"][:10]

    history = st.session_state.get("fruit_history", [])
    if history:
        st.markdown('<p class="section-label">Session Log</p><p class="section-title">Recent predictions (this session only)</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
        if st.button("🗑️ Clear history"):
            st.session_state["fruit_history"] = []
            st.rerun()


def render_model_info(classes):
    st.markdown('<p class="section-label" style="margin-top:0;">Model Info</p><p class="section-title">How this model was built and how it performs</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Backbone", "EfficientNetV2B0")
    m2.metric("Test Accuracy", "100%")
    m3.metric("Classes", str(len(classes)))
    m4.metric("Training", "2-phase fine-tune")

    st.markdown('<p class="section-label">Classes</p><p class="section-title">What this model recognizes</p>', unsafe_allow_html=True)
    cols = st.columns(len(classes))
    for col, c in zip(cols, classes):
        with col:
            st.markdown(f"""
            <div class="class-card">
                <div class="emoji">{CLASS_EMOJI.get(c, '🍏')}</div>
                <div class="name">{c}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Training Curves</p><p class="section-title">Accuracy & loss over training</p>', unsafe_allow_html=True)
    try:
        st.image(TRAINING_CURVES_URL, width="stretch", caption="Training/validation accuracy and loss curves")
    except Exception:
        st.info("Training curve chart is unavailable in this deployment.")

    with st.expander("🏗️ Architecture & training methodology", expanded=False):
        st.markdown("""
- **Backbone:** EfficientNetV2B0, pretrained on ImageNet, used for transfer learning.
- **Two-phase training:** first the classification head is trained with the backbone frozen,
  then the top layers are unfrozen and fine-tuned at a low learning rate.
- **Output:** softmax over 4 classes, so every prediction comes with a full confidence
  distribution rather than just a single label.
- **Why background-diversified data?** Standard fruit datasets (e.g. Fruits-360) are collected
  on plain white studio backgrounds. A model trained directly on that data tends to overfit
  to the background rather than the fruit, and quietly fails on ordinary real-world photos.
""")

    with st.expander("🌄 Why background augmentation?", expanded=False):
        st.markdown("""
This project explicitly targets that failure mode with a background-diversification pipeline:

1. **Background removal** from training images (rembg / U2-Net).
2. **Synthetic re-compositing** of the isolated fruit onto diverse backgrounds — solid colors,
   gradients, noise, wood textures, kitchen-like surfaces.
3. **Rotation, brightness/contrast, and blur augmentation** on top of that, so the model learns
   to key on the fruit itself, not the scene around it.
""")


def render_about(classes):
    st.markdown('<p class="section-label" style="margin-top:0;">About</p><p class="section-title">Project overview & limitations</p>', unsafe_allow_html=True)
    st.markdown(f"""
The Fruit AI Classifier is a deep learning image classifier that identifies **{', '.join(classes)}**
from photos, built with transfer learning on EfficientNetV2B0 and deployed as this interactive
Streamlit dashboard.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label" style="margin-top:0;">Key Facts</p>', unsafe_allow_html=True)
        st.markdown(f"""
- **Model:** EfficientNetV2B0 (transfer learning, fine-tuned)
- **Classes:** {', '.join(classes)}
- **Test accuracy:** 100% on the held-out test split
- **Generalization focus:** background-diversification augmentation pipeline
""")
    with c2:
        st.markdown('<p class="section-label" style="margin-top:0;">Limitations</p>', unsafe_allow_html=True)
        st.markdown("""
- Trained on exactly 4 fruit classes — any other object or fruit will still be forced into
  one of these 4 labels, so an out-of-scope photo can produce a confident but wrong answer.
- 100% held-out test accuracy reflects this specific test split, not a guarantee on every
  possible real-world photo.
- Best results come from photos where the fruit is reasonably visible and unobstructed.
""")

    st.markdown(f"""
    <div class="footer-note">
        Built by Muhammad Huzaifa Shamsi ·
        <a href="{GITHUB_URL}" target="_blank">Source on GitHub</a> ·
        <a href="{PORTFOLIO_URL}" target="_blank">Portfolio</a>
    </div>
    """, unsafe_allow_html=True)


def main():
    try:
        classifier, classes = load_model()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the model: {exc}")
        st.stop()
        return

    render_hero(classes)
    mode = render_sidebar_nav(classes)

    if mode == "classify":
        render_classify(classifier, classes)
    elif mode == "insights":
        render_model_info(classes)
    else:
        render_about(classes)


if __name__ == "__main__":
    main()
