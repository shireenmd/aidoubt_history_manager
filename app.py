import streamlit as st
import sqlite3
import os
import time
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Doubt History Manager",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found.")
    st.info("Please check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "doubt_history.db"


def create_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doubts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


create_database()


# ============================================================
# SAVE DOUBT
# ============================================================

def save_doubt(question, answer, category, difficulty):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO doubts
        (question, answer, category, difficulty, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        question,
        answer,
        category,
        difficulty,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# ============================================================
# GET DOUBTS
# ============================================================

def get_doubts():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        "SELECT * FROM doubts ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


# ============================================================
# DELETE DOUBT
# ============================================================

def delete_doubt(doubt_id):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM doubts WHERE id = ?",
        (doubt_id,)
    )

    conn.commit()
    conn.close()


# ============================================================
# AI FUNCTION
# ============================================================

def ask_ai(question):

    prompt = f"""
You are an AI study assistant.

The student has asked this question:

{question}

Explain the answer in a simple way suitable for a beginner.

Use exactly this structure:

1. Simple Explanation
Explain the concept in easy words.

2. Step-by-Step Explanation
Explain the concept step by step.

3. Easy Example
Give one simple example.

4. Key Point to Remember
Give one important point that the student should remember.

Avoid unnecessary complexity.
"""

    # Try the model 3 times if there is a temporary error
    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            if response.text:

                return response.text

            return "⚠️ AI returned an empty response."

        except Exception as e:

            error_message = str(e)

            # Temporary Gemini server errors
            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
            ):

                if attempt < 2:

                    time.sleep(5)

                    continue

            return f"❌ AI Error: {error_message}"

    return (
        "⚠️ Gemini AI is temporarily unavailable. "
        "Please try again after a few seconds."
    )


# ============================================================
# FIND SIMILAR DOUBTS
# ============================================================

def find_similar_doubts(question, df):

    if df.empty:
        return pd.DataFrame()

    if len(df) < 2:
        return pd.DataFrame()

    questions = df["question"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:

        vectors = vectorizer.fit_transform(
            questions + [question]
        )

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        result = df.copy()

        result["similarity"] = similarity

        result = result[
            result["similarity"] >= 0.10
        ]

        result = result.sort_values(
            by="similarity",
            ascending=False
        )

        return result.head(5)

    except Exception:

        return pd.DataFrame()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Doubt History Manager</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask doubts • Get AI answers • Save history • Track learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 Menu")

page = st.sidebar.radio(
    "Select a section",
    [
        "🏠 Ask Doubt",
        "📖 Doubt History",
        "🔎 Search Doubts",
        "📊 Analytics"
    ]
)


# ============================================================
# ASK DOUBT
# ============================================================

if page == "🏠 Ask Doubt":

    st.header("💬 Ask Your Doubt")

    category = st.selectbox(
        "📚 Select Subject",
        [
            "Python",
            "Java",
            "C Programming",
            "Data Structures",
            "DBMS",
            "Artificial Intelligence",
            "Machine Learning",
            "Software Engineering",
            "Other"
        ]
    )

    difficulty = st.selectbox(
        "📈 Difficulty Level",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )

    question = st.text_area(
        "✏️ Enter your doubt",
        placeholder="Example: What is inheritance in Java?",
        height=150
    )

    if st.button(
        "🤖 Ask AI",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "⚠️ Please enter your doubt first."
            )

        else:

            with st.spinner(
                "🤖 AI is thinking..."
            ):

                answer = ask_ai(question)

            if answer.startswith("❌ AI Error"):

                st.error(answer)

            elif answer.startswith("⚠️ Gemini"):

                st.warning(answer)

            else:

                st.success(
                    "✅ Answer generated successfully!"
                )

                st.markdown("## 📖 AI Answer")

                st.markdown(answer)

                # Save to database
                save_doubt(
                    question,
                    answer,
                    category,
                    difficulty
                )

                st.success(
                    "💾 Your doubt has been saved to history!"
                )

                # Find similar previous doubts
                all_doubts = get_doubts()

                if len(all_doubts) > 1:

                    previous_doubts = all_doubts.iloc[1:]

                    similar = find_similar_doubts(
                        question,
                        previous_doubts
                    )

                    if not similar.empty:

                        st.markdown(
                            "## 🔎 Similar Previous Doubts"
                        )

                        for _, row in similar.iterrows():

                            st.info(
                                f"**{row['question']}**"
                            )


# ============================================================
# DOUBT HISTORY
# ============================================================

elif page == "📖 Doubt History":

    st.header("📖 Doubt History")

    df = get_doubts()

    if df.empty:

        st.info(
            "📭 No doubts found. Ask your first doubt!"
        )

    else:

        st.metric(
            "Total Doubts",
            len(df)
        )

        st.markdown("---")

        for _, row in df.iterrows():

            title = (
                f"#{row['id']} | "
                f"{row['question']}"
            )

            with st.expander(title):

                st.write(
                    f"📚 **Subject:** {row['category']}"
                )

                st.write(
                    f"📈 **Difficulty:** {row['difficulty']}"
                )

                st.write(
                    f"🕒 **Date:** {row['created_at']}"
                )

                st.markdown("---")

                st.markdown("### 🤖 AI Answer")

                st.markdown(row["answer"])

                if st.button(
                    "🗑️ Delete Doubt",
                    key=f"delete_{row['id']}"
                ):

                    delete_doubt(row["id"])

                    st.success(
                        "Doubt deleted successfully!"
                    )

                    time.sleep(0.5)

                    st.rerun()

        st.markdown("---")

        # CSV DOWNLOAD
        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Doubt History",
            csv_data,
            "doubt_history.csv",
            "text/csv"
        )


# ============================================================
# SEARCH DOUBTS
# ============================================================

elif page == "🔎 Search Doubts":

    st.header("🔎 Search Your Doubts")

    df = get_doubts()

    if df.empty:

        st.info(
            "📭 No doubt history available."
        )

    else:

        search_text = st.text_input(
            "Search for a doubt",
            placeholder="Example: inheritance"
        )

        if search_text.strip():

            results = df[
                df["question"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

            if results.empty:

                st.warning(
                    "❌ No matching doubts found."
                )

            else:

                st.success(
                    f"Found {len(results)} matching doubt(s)."
                )

                for _, row in results.iterrows():

                    with st.expander(
                        row["question"]
                    ):

                        st.write(
                            f"📚 Subject: {row['category']}"
                        )

                        st.write(
                            f"📈 Difficulty: {row['difficulty']}"
                        )

                        st.write(
                            f"🕒 Date: {row['created_at']}"
                        )

                        st.markdown("---")

                        st.markdown(
                            row["answer"]
                        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.header("📊 Learning Analytics")

    df = get_doubts()

    if df.empty:

        st.info(
            "📭 Ask some doubts to see your analytics."
        )

    else:

        total = len(df)

        easy = len(
            df[df["difficulty"] == "Easy"]
        )

        medium = len(
            df[df["difficulty"] == "Medium"]
        )

        hard = len(
            df[df["difficulty"] == "Hard"]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "📚 Total",
            total
        )

        col2.metric(
            "🟢 Easy",
            easy
        )

        col3.metric(
            "🟡 Medium",
            medium
        )

        col4.metric(
            "🔴 Hard",
            hard
        )

        st.markdown("---")

        # SUBJECT ANALYTICS
        st.subheader(
            "📚 Doubts by Subject"
        )

        subject_counts = df[
            "category"
        ].value_counts()

        st.bar_chart(
            subject_counts
        )

        # DIFFICULTY ANALYTICS
        st.subheader(
            "📈 Difficulty Distribution"
        )

        difficulty_counts = df[
            "difficulty"
        ].value_counts()

        st.bar_chart(
            difficulty_counts
        )

        # RECENT DOUBTS
        st.subheader(
            "🕒 Recent Doubts"
        )

        recent = df.head(5)

        st.dataframe(
            recent[
                [
                    "question",
                    "category",
                    "difficulty",
                    "created_at"
                ]
            ],
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🤖 AI Doubt History Manager | "
    "Python + Streamlit + Gemini AI + SQLite + Scikit-learn"
)