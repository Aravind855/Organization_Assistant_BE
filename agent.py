from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from llm import get_llm

def get_productivity_agent(tools: list = None):
    """
    Creates and returns a LangChain agent using create_agent with Gemini model.
    """
    if tools is None:
        tools = []
        
    model = get_llm()
    
    system_prompt = (
        "You are an AI-powered Productivity Assistant. "
        "You interact with managers, team leads, HR professionals, and employees "
        "to assist them with Gmail and Google Calendar workflows. "
        "Be helpful, structured, clear, and professional. "
        "Use available tools when necessary to perform email and calendar operations."
    )
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )
    
    return agent
