from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agent import get_productivity_agent

app = FastAPI(
    title="Productivity Assistant API",
    description="AI-powered Gmail & Google Calendar Assistant API using LangChain Agent & Gemini",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: str

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default_session"
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    reply: str
    status: str = "success"

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Productivity Assistant Agent API powered by LangChain and Google Gemini is running"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        agent = get_productivity_agent()
        
        # Invoke agent with thread_id for conversation persistence
        config = {"configurable": {"thread_id": request.thread_id or "default_session"}}
        
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config=config,
        )
        
        # Extract response message from agent state
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            reply = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            reply = "No response generated."
            
        return ChatResponse(reply=str(reply))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error executing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
