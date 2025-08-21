## 📝 AI‑Конспектор текста (Streamlit)

Веб‑приложение, которое сжимает длинный текст в краткий конспект. Работает с провайдерами: OpenAI (по умолчанию) и Ollama (локально, LLaMA2/LLaMA3 и др.).

### Возможности
- Ввод текста или загрузка файлов `.txt/.pdf`
- Режимы длины: короткий / средний / длинный
- Формат: маркированные пункты или связный текст
- Язык: авто (как вход), русский, английский
- Провайдер: OpenAI (gpt-4o-mini и др.) или Ollama (llama2/llama3/mistral)

### Установка
1) Перейдите в папку проекта:

```bash
cd "your project"
```

2) Создайте и активируйте виртуальное окружение (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Установите зависимости:

```bash
pip install -r requirements.txt
```

4) Скопируйте пример переменных окружения и пропишите ключ:

```bash
Copy-Item .env.example .env
# Откройте .env и вставьте ваш ключ OpenAI
```

### Переменные окружения
Скопируйте `your project/.env.example` в `.env` и заполните при необходимости:

```env
OPENAI_API_KEY=sk-...     # ключ OpenAI (если используете OpenAI)
OLLAMA_BASE_URL=http://localhost:11434  # адрес Ollama (для локальных моделей)
OLLAMA_MODEL=llama2
```

### Запуск

```bash
streamlit run "your project/Project.py"
```

Приложение откроется в браузере. Выберите провайдера, модель и введите текст.

### Ollama (локально, LLaMA2)
- Установите Ollama: `https://ollama.com`
- Скачайте модель: `ollama pull llama2` (или `llama3`)
- Запустите сервис (обычно запускается автоматически), проверьте `http://localhost:11434`
- В интерфейсе выберите провайдер `Ollama` и модель `llama2`

### Стек
- Python, Streamlit
- LangChain (`langchain`, `langchain-openai`, `langchain-ollama`)
- OpenAI API или локальная Ollama (LLaMA2/3)
- pypdf для чтения PDF

