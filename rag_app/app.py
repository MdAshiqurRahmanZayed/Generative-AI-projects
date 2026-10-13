import streamlit as st
from dotenv import load_dotenv

import rag_core

load_dotenv()

st.set_page_config(page_title="RAG Knowledge Base", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 RAG Knowledge Base")

with st.sidebar:
    st.header("Add a source")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "docx", "txt", "md"],
        key="uploader",
    )
    if uploaded_file is not None:
        if st.button("Add file", use_container_width=True):
            try:
                rag_core.add_file_source(uploaded_file.name, uploaded_file.getvalue())
                st.success(f"Added: {uploaded_file.name}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.divider()

    link_url = st.text_input("Or add a link", placeholder="https://example.com/article")
    if st.button("Add link", use_container_width=True):
        if link_url:
            try:
                rag_core.add_link_source(link_url)
                st.success(f"Added: {link_url}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.warning("Enter a URL first.")

    st.divider()

    sources = rag_core.load_manifest()
    pending_count = sum(1 for s in sources if s.status != "synced")

    st.header(f"Sources ({len(sources)})")

    if pending_count > 0:
        if st.button(f"🔄 Sync ({pending_count} pending)", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Starting sync...")

            def _on_progress(i, total, source):
                progress_bar.progress((i) / total, text=f"Embedding: {source.name}")

            results = rag_core.sync_pending(progress_callback=_on_progress)
            progress_bar.progress(1.0, text="Done")

            errors = [r for r in results if r.status == "error"]
            if errors:
                st.error(f"{len(errors)} source(s) failed to sync. See list below.")
            else:
                st.success(f"Synced {len(results)} source(s).")
            st.rerun()
    else:
        st.caption("Everything is synced.")

    status_icon = {"synced": "✅", "pending": "⏳", "error": "❌"}
    for s in sources:
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{status_icon.get(s.status, '?')} {s.name}"
            if s.status == "error":
                st.markdown(f"{label}")
                st.caption(s.error)
            else:
                st.markdown(label)
        with col2:
            if st.button("🗑️", key=f"del_{s.id}", help="Remove"):
                rag_core.remove_source(s.id)
                st.rerun()

st.caption(
    "Ask questions about your synced files/links. "
    "If the answer isn't in your data, I'll say so."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, _docs = rag_core.ask(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
