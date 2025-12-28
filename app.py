import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic import hub
import os
from dotenv import load_dotenv

load_dotenv()

## Page configuration
st.title("🔎 LangChain - Chat with Search")


# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize tools
@st.cache_resource
def get_tools():
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    
    search = DuckDuckGoSearchRun(name="Search")
    
    return [search, arxiv, wiki]

tools = get_tools()

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Chat input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    if not api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )
    
    # Pull the ReAct prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")
    
    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    # Display assistant response with callback handler
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(
            parent_container=st.container(),
            expand_new_thoughts=True,
            collapse_completed_thoughts=True
        )
        
        try:
            # Invoke the agent with the callback
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            
            # Get the final output
            final_answer = response.get("output", "I couldn't generate a response.")
            
            # Display the answer
            st.write(final_answer)
            
            # Add to chat history
            st.session_state.messages.append({
                'role': 'assistant',
                "content": final_answer
            })
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                'role': 'assistant',
                "content": error_msg
            })
