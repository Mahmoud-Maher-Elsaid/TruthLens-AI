"""Small internship lab client. The production interface is the Next.js app."""
import os
import requests
import streamlit as st

st.set_page_config(page_title="TruthLens AI Lab", page_icon="🔎")
st.title("TruthLens AI — Streamlit Lab")
st.caption("Educational client for the shared FastAPI backend. Production UI: Next.js/Vercel.")
api = os.getenv("TRUTHLENS_API_BASE_URL", "http://localhost:8000").rstrip("/")
text = st.text_area("AI response or claim", height=220)
mode = st.selectbox("Mode", ["demo", "standard", "deep"])
if st.button("Audit", type="primary"):
    if len(text.strip()) < 3: st.error("Enter a factual statement.")
    else:
        try:
            response = requests.post(f"{api}/api/v1/analyze", json={"text": text, "mode": mode}, timeout=120); response.raise_for_status(); data = response.json()
            if data.get("is_precomputed_demo"): st.warning("Precomputed deterministic demo — not fresh model inference.")
            st.metric("Claims analyzed", data["summary"]["claims_analyzed"])
            for claim in data["claims"]:
                with st.expander(f"{claim['verdict']} · {claim['claim']}"): st.write(claim["explanation"]); st.json(claim["evidence"])
        except requests.RequestException as exc: st.error(f"Backend unavailable: {exc}")
