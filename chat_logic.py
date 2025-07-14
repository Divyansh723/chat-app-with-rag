from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
import sqlite3
from typing import Literal
from dotenv import load_dotenv
import os
from fpdf import FPDF
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

# Load environment variables
load_dotenv()

# --- Tools ---

def search_web(query: str) -> str:
    
    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query=query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return formatted_search_docs

def search_wikipedia(query:str) -> str:
    
    """ Retrieve docs from wikipedia """

    # Search
    search_docs = WikipediaLoader(query=query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return formatted_search_docs

def add(a: int, b: int) -> int:
    """
    Tool: Adds two integers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: Sum of a and b.
    """
    print("Adding numbers:", a, b)
    return a + b

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persistent Chroma vector DB
vectordb = Chroma(
    persist_directory="chroma_db/my_docs",
    embedding_function=embedding_model,
    collection_name="my_docs"
)

# @tool(name="db_search", description="Searches uploaded documents for relevant information using semantic vector similarity. Provide a natural language question.")
def db_search(query: str) -> str:
    """
    Semantic search in vector DB. Use this to answer queries based on uploaded documents.
    If no match is found, fallback to general knowledge.
    """
    print("abc")
    try:
        query_vector = embedding_model.embed_query(query)
        results = vectordb.similarity_search_by_vector(query_vector, k=3)
        if not results:
            return "⚠️ No relevant content found in uploaded documents."
        combined = "\n\n".join([doc.page_content for doc in results])
        return f"📄 Relevant context from documents:\n\n{combined}"
    except Exception as e:
        return f"❌ Error in retrieval: {str(e)}"

# --- LangGraph State ---

class State(MessagesState):
    summary: str
    token_count: int = 0

# Token estimator
def estimate_tokens(text: str) -> int:
    return len(text) // 4

# Model call logic
def call_model(state: State):
    summary = state.get("summary", "")
    token_count = state.get("token_count", 0)

    messages = [SystemMessage(content=f"Summary of conversation earlier: {summary} remember you have access to various tools to use so use them when ever they are needed to complete the task")] + state["messages"] if summary else state["messages"]
    total_new_tokens = sum(estimate_tokens(m.content) for m in messages if hasattr(m, "content"))

    response = model.invoke(messages)
    return {"messages": response, "token_count": token_count + total_new_tokens}

# Summarization condition
def should_continue(state: State) -> Literal["summarize_conversation", "__end__"]:
    return "summarize_conversation" if state.get("token_count", 0) > 1000 else END

# Summarization logic
def summarize_conversation(state: State):
    summary = state.get("summary", "")
    summary_msg = (
        f"This is summary of the conversation to date: {summary}\n\nExtend the summary by taking into account the new messages above:"
        if summary else "Create a summary of the conversation above:"
    )
    messages = state["messages"] + [HumanMessage(content=summary_msg)]
    response = model.invoke(messages)

    if estimate_tokens(response.content) > 1000:
        compressed = model.invoke([HumanMessage(content="Compress the following summary:\n\n" + response.content)])
        response = compressed

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages, "token_count": 0}

# Load Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_output_tokens=1000
)

# Bind tools to Gemini
tools = [add, db_search, search_web, search_wikipedia]
model.bind_tools(tools)

# SQLite-based checkpointer
session_conn = sqlite3.connect(
    r"D:\langchain\langchain-academy\module-2\state_db\session_and_history.db",
    check_same_thread=False
)
session = SqliteSaver(conn=session_conn)

# LangGraph flow setup
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", tools_condition)
workflow.add_edge("tools", "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

graph = workflow.compile(checkpointer=session)
config = {"configurable": {"thread_id": "3"}}

# --- App Interface ---

def ask_tempest(text):
    """
    Adds a system prompt, runs query through LangGraph, and returns response and summary.
    """ 
    system_prompt = (
        "You are Tempest, a helpful and intelligent assistant. "
        # "You have access to a tool called `db_search`. "
        # "When a user asks a question, if it could be related to uploaded documents, call the `db_search` tool with that question. "
        # "Use only the question as the input to the tool. "
        # "Do NOT return code. Do NOT explain the tool call. Simply use the tool directly. "
        # "Only fall back to your own knowledge if the tool returns no relevant results. "
        # "Be clear, direct, and helpful in your answers."
    )

    system_message = SystemMessage(content=system_prompt)
    user_message = HumanMessage(content=text)

    result = graph.invoke({"messages": [system_message, user_message]}, config)
    response = result["messages"][-1].content if result["messages"] else ""
    summary = graph.get_state(config).values.get("summary", "")
    return response, summary

def save_chat(chat, session_id, as_pdf=False):
    os.makedirs("chat_logs", exist_ok=True)
    if as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for sender, msg in chat:
            pdf.multi_cell(0, 10, f"{sender}: {msg}")
        pdf.output(f"chat_logs/{session_id}.pdf")
    else:
        with open(f"chat_logs/{session_id}.txt", "w", encoding="utf-8") as f:
            for sender, msg in chat:
                f.write(f"{sender}: {msg}\n")

def load_chat(session_id):
    path = f"chat_logs/{session_id}.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [(line.split(":")[0], ":".join(line.split(":")[1:]).strip()) for line in lines]
    return []
