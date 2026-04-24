import json
import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"
CHAT_TEMPERATURE = 0.4

with open("askfirst_synthetic_dataset.json", "r") as f:
    dataset = json.load(f)

SYSTEM_PROMPT = """You are a medical pattern analyst for Ask First, a health clarity platform.

Your job is to find ALL hidden cross-conversation health patterns with causal temporal reasoning.

CRITICAL: There are approximately 8 patterns hidden across 3 users. Find all of them. Do not stop after 2.

TYPES OF PATTERNS TO LOOK FOR:
- Symptom appearing WEEKS after a lifestyle change (not days) — e.g. hair fall 6-8 weeks after calorie restriction started
- Symptoms that COMPOUND over time from a single root cause appearing in sequence
- Symptoms that RESOLVE when a habit changes — resolution confirms causation
- Cyclical patterns tied to monthly events (menstrual cycle, work sprints recurring month after month)
- One root habit causing MULTIPLE downstream symptoms across different sessions
- Dose-response relationships (more dairy = more acne, less dairy = clear skin)

TEMPORAL REASONING RULES:
1. Calculate actual week gaps between sessions using timestamps. State them explicitly in numbers.
2. Hair fall 6-8 weeks after nutritional deficiency = telogen effluvium. Not coincidence.
3. Symptom resolving after habit stops = causal confirmation. Cite the resolution session.
4. Same symptom recurring each month with same preceding condition = cyclical causal pattern.
5. Absence of symptom in sessions without the trigger = strong negative evidence. Always cite it.

STRICT OUTPUT RULES:
- Output ONLY a valid JSON array. No markdown fences, no preamble, no text outside the JSON.
- Exhaustively analyze every session of every user before concluding.
- Do not truncate. Return every pattern found.

Each pattern object must have exactly these fields:
{
  "pattern_id": "P1",
  "user": "<user_id e.g. USR001>",
  "title": "<short descriptive title>",
  "sessions_involved": ["S01", "S04"],
  "temporal_reasoning": "<explicit week gap calculation, causal mechanism, why timing matters>",
  "reasoning_trace": "<what you ruled out, negative evidence from sessions without the symptom>",
  "confidence": "high | medium | low",
  "confidence_justification": "<one line with session count, consistency, and dose-response evidence if any>"
}"""


def build_user_context(user: dict) -> str:
    lines = [
        f"USER PROFILE: {user['name']}, age {user['age']}, {user['occupation']}, {user['location']}",
        f"ONBOARDING: {user['onboarding_notes']}",
        "---CONVERSATION HISTORY---"
    ]
    for c in user["conversations"]:
        session_num = c["session_id"].split("_")[1]
        lines.append(f"\n[{c['timestamp']}] {session_num}")
        lines.append(f"User: {c['user_message']}")
        if c.get("user_followup"):
            lines.append(f"User followup: {c['user_followup']}")
        lines.append(f"Clary: {c['clary_response']}")
        lines.append(f"Tags: {', '.join(c.get('tags', []))}")
    return "\n".join(lines)


def calibrate_confidence(pattern: dict) -> dict:
    sessions = pattern.get("sessions_involved", [])
    if len(sessions) < 3 and pattern.get("confidence") == "high":
        pattern["confidence"] = "medium"
        pattern["confidence_justification"] += " [calibrated: <3 sessions]"
    return pattern


def parse_json(raw: str) -> list:
    clean = raw.strip()
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["):
                clean = part
                break
    return json.loads(clean)


def run_analysis(user: dict, trace_box, status_box):
    context = build_user_context(user)
    prompt = f"""Analyze this user's complete health history across ALL sessions.

{context}

IMPORTANT: Find EVERY pattern, not just the most obvious ones. Look for:
- Delayed symptom onset weeks after a trigger
- Symptoms that resolved when a habit changed (resolution = causal proof)
- Multiple symptoms from one root cause appearing in sequence over weeks
- Monthly recurring patterns tied to the same condition each cycle

Return only a JSON array. No text outside the JSON."""

    full_response = ""
    for attempt in range(1, 4):
        full_response = ""
        status_box.info(f"Attempt {attempt}/3 — streaming from OpenAI...")
        try:
            with client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        trace_box.code(full_response[-3000:], language="json")

            patterns = parse_json(full_response)
            status_box.empty()
            trace_box.empty()
            return [calibrate_confidence(p) for p in patterns] if isinstance(patterns, list) else []

        except json.JSONDecodeError:
            status_box.warning(f"JSON parse failed on attempt {attempt}. Retrying...")
            time.sleep(1)
        except Exception as e:
            status_box.error(f"API error: {e}")
            return []

    status_box.error("Failed after 3 attempts. Raw output shown above.")
    return []


def confidence_badge(level: str) -> str:
    return {"high": "🟢 HIGH", "medium": "🟡 MEDIUM", "low": "🔴 LOW"}.get(level, "⚪ UNKNOWN")


# --- Streamlit UI ---

st.set_page_config(page_title="Ask First — Clary", layout="wide")
st.title("🧠 Ask First — Clary")
st.caption("Cross-conversation health pattern detection with temporal causal reasoning")

user_map = {u["name"]: u for u in dataset["users"]}
selected_name = st.selectbox("Select User", list(user_map.keys()))
user = user_map[selected_name]

col1, col2, col3 = st.columns(3)
col1.metric("Name", user["name"])
col2.metric("Age / Gender", f"{user['age']} / {user['gender']}")
col3.metric("Sessions", len(user["conversations"]))
st.caption(f"**Onboarding notes:** {user['onboarding_notes']}")

tab_chat, tab_patterns = st.tabs(["💬 Chat with Clary", "🔍 Pattern Analysis"])

# ── Chat tab ──────────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are Clary, a compassionate and knowledgeable health clarity assistant for Ask First.

You have access to the user's full health conversation history below. Use it to give
personalized, context-aware answers. When relevant, reason about timing, triggers, and
patterns across sessions. Speak in a warm, clear, non-alarmist tone.

{context}"""

with tab_chat:
    # Session-state key includes the selected user so history resets on user change
    history_key = f"chat_history_{selected_name}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Render existing messages
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask Clary anything about your health…")
    if user_input:
        # Show and store user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state[history_key].append({"role": "user", "content": user_input})

        # Build messages list for the API
        system_content = CHAT_SYSTEM_PROMPT.format(context=build_user_context(user))
        api_messages = [{"role": "system", "content": system_content}] + st.session_state[history_key]

        # Stream the assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_reply = ""
            try:
                with client.chat.completions.create(
                    model=MODEL,
                    messages=api_messages,
                    temperature=CHAT_TEMPERATURE,
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
                        if delta:
                            full_reply += delta
                            response_placeholder.markdown(full_reply + "▌")
                response_placeholder.markdown(full_reply)
            except Exception as e:
                full_reply = f"⚠️ Sorry, something went wrong: {e}"
                response_placeholder.markdown(full_reply)

        st.session_state[history_key].append({"role": "assistant", "content": full_reply})

    if history_key in st.session_state and st.session_state[history_key]:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state[history_key] = []
            st.rerun()

# ── Pattern Analysis tab ───────────────────────────────────────────────────────

with tab_patterns:
    if st.button("🔍 Analyze Patterns", type="primary"):
        st.subheader("Reasoning Trace (live stream)")
        trace_box = st.empty()
        status_box = st.empty()

        patterns = run_analysis(user, trace_box, status_box)

        if patterns:
            st.success(f"✅ Found {len(patterns)} pattern(s)  |  Expected: 8  |  Coverage: {len(patterns)}/8")
            for p in patterns:
                with st.expander(f"{confidence_badge(p.get('confidence', ''))}  —  {p.get('title', 'Untitled')}"):
                    st.markdown(f"**Sessions involved:** `{', '.join(p.get('sessions_involved', []))}`")
                    st.markdown("**Temporal Reasoning**")
                    st.write(p.get("temporal_reasoning", ""))
                    st.markdown("**Reasoning Trace**")
                    st.write(p.get("reasoning_trace", ""))
                    st.info(f"**Confidence justification:** {p.get('confidence_justification', '')}")

            with st.expander("📄 Raw JSON output"):
                st.json(patterns)

            st.download_button(
                "⬇️ Download patterns_output.json",
                data=json.dumps(patterns, indent=2),
                file_name="patterns_output.json",
                mime="application/json"
            )
        else:
            st.warning("No patterns extracted.")