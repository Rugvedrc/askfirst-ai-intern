**README.md**

```markdown
# Ask First — Clary Pattern Detection

AI-powered cross-conversation health pattern detection with temporal causal reasoning.

## Setup

```bash
pip install openai streamlit python-dotenv
```

Add your key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

## Screenshots

**Arjun (USR001)**
![Arjun](arjun.png)

**Meera (USR002)**
![Meera](meera.png)

**Priya (USR003)**
![Priya](priya.png)

## How it works

- Full conversation history sent in one shot per user (no chunking — preserves temporal context)
- GPT-4o reasons across timestamps to find delayed, cyclical, and compounding patterns
- Live reasoning trace streamed to UI before final JSON output
- Confidence auto-downgraded to `medium` if fewer than 3 sessions support a pattern
- JSON parse retries up to 3 times on failure

## Model

**gpt-4o** — strong structured JSON output, large context window, native streaming.

## Project Structure

```
├── app.py
├── askfirst_synthetic_dataset.json
├── one_page_writeup.md
├── .env               # not committed
├── arjun.png
├── meera.png
├── priya.png
├── .gitignore
└── README.md
```


