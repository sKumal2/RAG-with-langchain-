import os
# from vector_store import vector_store
from indexing import vector_store
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import uuid7
from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage
load_dotenv()


model =ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite" )

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(
    model= model,
    tools= tools,
    system_prompt=prompt)


query = (
    "Give Deatils on MIPS."
)
       
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()