# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from pathlib import Path
import uvicorn

# Import main function as process_magical_query from test.py
from test import main as process_magical_query

# Initialize FastAPI app
app = FastAPI(
    title="Magical Knowledge API",
    description="A web API for accessing magical knowledge about spells and potions",
    version="1.0.0",
)

# CORS settings- allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and response models for the API
class MagicalRequest(BaseModel):
    query: str


class MagicalResponse(BaseModel):
    answer: str
    sources: List[str]

# Root confirmation endpoint
@app.get("/")
async def root():
    """Welcome endpoint that confirms the API is running."""
    return {
        "status": "active",
        "message": "Welcome to the Magical Knowledge API",
        "endpoints": {
            "/query": "POST endpoint for magical queries",
            "/docs": "Interactive API documentation",
        },
    }

# Main endpoint- access this from frontend
@app.post("/query", response_model=MagicalResponse)
async def get_magical_knowledge(request: MagicalRequest):
    """
    Process a magical query and return relevant information.

    Args:
        request: MagicalRequest containing the query string

    Returns:
        MagicalResponse containing the answer and sources
    """
    try:
        # Call the main function (alias is process_magical_query) from test.py
        result = process_magical_query(request.query)

        # Return the result as a dictionary
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        # Error handling in case of any exceptions
        raise HTTPException(
            status_code=500, detail=f"Error processing magical query: {str(e)}"
        )


# Run the application automatically using command: python main.py
if __name__ == "__main__":
    port = 8000 # Local host port; make sure nothing is running on this port
    uvicorn.run(app, host="0.0.0.0", port=port)
