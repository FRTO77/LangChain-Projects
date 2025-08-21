import os
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from a local .env file if present."""
    try:
        load_dotenv()
    except Exception:
        pass


def read_text_from_file(uploaded_file) -> str:
    """Read text from a Streamlit uploaded file (.txt or .pdf)."""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()
    if filename.endswith(".txt"):
        raw_bytes = uploaded_file.read()
        try:
            return raw_bytes.decode("utf-8")
        except Exception:
            return raw_bytes.decode("latin-1", errors="ignore")

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception as exc:
            st.error("Для чтения PDF требуется зависимость 'pypdf'. Добавьте её в окружение.")
            raise exc

        reader = PdfReader(uploaded_file)
        pages_text: List[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        return "\n\n".join(pages_text)

    st.warning("Поддерживаются только файлы .txt и .pdf")
    return ""


def make_llm(provider: str, model: str, api_key: Optional[str], temperature: float = 0.2):
    """Create an LLM instance for the chosen provider."""
    if provider == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            st.error("Не найдена библиотека 'langchain-openai'. Установите зависимости из requirements.txt")
            raise exc

        effective_key = api_key or os.getenv("OPENAI_API_KEY")
        if not effective_key:
            st.stop()
        return ChatOpenAI(model=model, api_key=effective_key, temperature=temperature)

    if provider == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception as exc:
            st.error("Не найдена библиотека 'langchain-ollama'. Установите зависимости из requirements.txt")
            raise exc

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    raise ValueError(f"Unknown provider: {provider}")


def chunk_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """Split long text into chunks using LangChain's RecursiveCharacterTextSplitter if available.
    Fallback to a simple splitter by characters.
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".", "? ", "! ", ", ", ",", " "]
        )
        docs = splitter.create_documents([text])
        return [d.page_content for d in docs]
    except Exception:
        # Simple fallback: naive split by characters with safe stepping
        chunks: List[str] = []
        n = len(text)
        safe_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        step = max(1, chunk_size - safe_overlap)
        i = 0
        while i < n:
            end = min(i + chunk_size, n)
            chunks.append(text[i:end])
            if end >= n:
                break
            i += step
        return chunks


def build_chunk_prompt(
    chunk: str,
    target_length: str,
    bullet_points: bool,
    language_pref: str,
) -> str:
    """Prompt to summarize a single chunk."""
    formatting = (
        "Сформируй маркированный список из 5-10 пунктов" if bullet_points else "Сформируй связный абзац из 5-8 предложений"
    )
    language_instruction = (
        "Ответь на том же языке, что и входной текст." if language_pref == "Авто" else f"Ответь на {language_pref}."
    )

    return (
        f"Ты — эксперт по конспектированию. Сожми следующий текст в {target_length} конспект для занятого читателя.\n\n"
        f"Требования:\n"
        f"- {formatting}\n"
        f"- {language_instruction}\n"
        f"- Сохраняй ключевые факты, цифры, имена, причинно-следственные связи\n"
        f"- Избегай воды и повторов, не придумывай новых фактов\n\n"
        f"Текст:\n{chunk}\n"
    )


def build_combine_prompt(
    partial_summaries: str,
    target_length: str,
    bullet_points: bool,
    language_pref: str,
) -> str:
    formatting = (
        "Сформируй маркированный список из 5-12 пунктов" if bullet_points else "Сформируй связный абзац(ы) из 8-15 предложений"
    )
    language_instruction = (
        "Ответь на том же языке, что и входной текст." if language_pref == "Авто" else f"Ответь на {language_pref}."
    )

    return (
        f"Ты — эксперт по сжатию информации. Объедини частичные конспекты ниже в один цельный {target_length} конспект.\n\n"
        f"Требования:\n"
        f"- {formatting}\n"
        f"- {language_instruction}\n"
        f"- Сохраняй структуру и ключевые факты без повтора\n\n"
        f"Частичные конспекты:\n{partial_summaries}\n"
    )


def call_llm(llm, prompt: str) -> str:
    """Call the chat model with a system+user style prompt packed into a single user message."""
    try:
        # Many LangChain chat models accept plain strings via .invoke
        result = llm.invoke(prompt)
        # For Chat models, content is on .content
        content = getattr(result, "content", None)
        return content if isinstance(content, str) and content.strip() else (str(result) if result else "")
    except Exception as exc:
        st.error(f"Ошибка вызова LLM: {exc}")
        raise


def summarize_long_text(
    llm,
    text: str,
    target_length: str,
    bullet_points: bool,
    language_pref: str,
) -> str:
    """Chunk the text, summarize each chunk, then combine."""
    chunks = chunk_text(text)
    if not chunks:
        return ""

    if len(chunks) == 1:
        single_prompt = build_chunk_prompt(chunks[0], target_length, bullet_points, language_pref)
        return call_llm(llm, single_prompt)

    partials: List[str] = []
    for idx, ch in enumerate(chunks, start=1):
        with st.spinner(f"Суммаризация фрагмента {idx}/{len(chunks)}…"):
            partials.append(call_llm(llm, build_chunk_prompt(ch, target_length, bullet_points, language_pref)))

    combined_prompt = build_combine_prompt("\n\n".join(partials), target_length, bullet_points, language_pref)
    return call_llm(llm, combined_prompt)


def main():
    load_env()

    st.set_page_config(page_title="AI‑Конспектор", page_icon="📝", layout="centered")
    st.title("📝 AI‑конспектор текста")
    st.caption("Python + LangChain + OpenAI/Ollama + Streamlit")

    with st.sidebar:
        st.header("Настройки")
        provider = st.selectbox("Провайдер", ["OpenAI", "Ollama"], index=0)

        if provider == "OpenAI":
            default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            model = st.selectbox("Модель (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
            api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        else:
            default_model = os.getenv("OLLAMA_MODEL", "llama2")
            model = st.text_input("Модель (Ollama)", value=default_model, help="Например: llama2, llama3, mistral")
            api_key = None

        target_length = st.radio(
            "Длина конспекта",
            options=["Короткий", "Средний", "Длинный"],
            index=1,
            help="Короткий ≈ 3–5 пунктов, Средний ≈ 6–10, Длинный ≈ 10–15"
        )
        bullet_points = st.toggle("Маркированные пункты", value=True)
        language_pref = st.selectbox("Язык вывода", ["Авто", "Русский", "English"], index=0)

    st.subheader("Входные данные")
    tab_text, tab_file = st.tabs(["Вставить текст", "Загрузить файл (.txt/.pdf)"])

    with tab_text:
        input_text = st.text_area(
            "Текст для конспекта",
            height=240,
            placeholder="Вставьте или напишите сюда длинный текст…",
        ).strip()

    with tab_file:
        uploaded = st.file_uploader("Выберите файл", type=["txt", "pdf"], accept_multiple_files=False)
        if uploaded is not None and not input_text:
            input_text = read_text_from_file(uploaded)

    if st.button("Сжать текст"):
        if not input_text:
            st.warning("Введите текст или загрузите файл.")
            st.stop()

        with st.spinner("Подготавливаем модель…"):
            llm = make_llm(provider=provider, model=model, api_key=api_key, temperature=0.2)

        with st.spinner("Генерируем конспект…"):
            summary = summarize_long_text(
                llm=llm,
                text=input_text,
                target_length=target_length,
                bullet_points=bullet_points,
                language_pref=language_pref,
            )

        if summary:
            st.success("Готово!")
            st.subheader("Результат")
            st.write(summary)
            st.download_button("⬇️ Скачать как TXT", data=summary, file_name="summary.txt")
        else:
            st.error("Не удалось получить конспект. Попробуйте ещё раз или измените настройки.")


if __name__ == "__main__":
    main()


