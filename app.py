import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Web Search")

def route_tools(query: str):
    query = query.lower()
    if any(word in query for word in ["paper", "research", "study", "arxiv"]):
        return [arxiv]
    elif any(word in query for word in ["history", "who is", "what is", "wikipedia"]):
        return [wiki]
    else:
        return [search]

st.title("🔍 AI Search Copilot (LangChain + Groq)")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I’m an AI search assistant. Ask me anything!"}
    ]

if not api_key:
    st.warning("Please enter your Groq API Key to start chatting.")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    tools = route_tools(prompt)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        handling_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = agent.run(
            f"""
            Answer the user query clearly and concisely.
            Cite the source used (Arxiv, Wikipedia, or Web) at the end.
            Query: {prompt}
            """,
            callbacks=[st_cb]
        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
