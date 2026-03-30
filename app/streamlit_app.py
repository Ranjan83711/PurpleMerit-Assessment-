import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from crew.crew_setup import CoursePlannerCrew

load_dotenv()

st.set_page_config(
    page_title="🎓 KUK Course Planner",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Agentic Course Planning Assistant")
st.markdown("""
Welcome to the Kurukshetra University (KUK) Course Planning Assistant.
I can help you:
- Check prerequisite eligibility
- Build a course plan for next semester
- Explain program requirements
- Answer academic policy questions
""")

st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Required to run the Assistant")

if "planner_crew" not in st.session_state:
    st.session_state.planner_crew = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if api_key and not st.session_state.planner_crew:
    try:
        st.session_state.planner_crew = CoursePlannerCrew(groq_api_key=api_key)
        st.sidebar.success("✅ System Initialized and Ready")
    except Exception as e:
        import traceback
        st.sidebar.error(f"Initialization Failed: {e}\n\nTraceback:\n{traceback.format_exc()}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about course planning or prerequisites (e.g. 'Can I take CS301 next semester? I have done CS101'):"):
    # React to user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
            response = "Error: Missing API Key."
        else:
            with st.spinner("Analyzing request, fetching documents, and reasoning the best course plan..."):
                try:
                    # Run the Crew AI Workflow
                    response = st.session_state.planner_crew.run(prompt)
                    # Support for CrewOutput which is an object in some versions of CrewAI
                    if hasattr(response, 'raw'):
                        response = response.raw
                    else:
                        response = str(response)
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    response = f"An error occurred: {e}"
        
    st.session_state.messages.append({"role": "assistant", "content": response})
