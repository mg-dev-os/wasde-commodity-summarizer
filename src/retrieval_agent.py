"""Microsoft Agent Framework (MAF) agent that performs retrieval via a vector store tool."""

import asyncio
from typing import Annotated, Any

from pydantic import Field

from src.config_loader import get_openai_api_key
from src.vector_store import search as vector_search


def create_search_tool(doc_id: str, filters: dict = None):
    """Create a search_document tool bound to the current document and optional filters."""

    def search_document(
        query: Annotated[
            str,
            Field(description="Natural language question or search query about the WASDE report (e.g. wheat exports)."),
        ],
    ) -> str:
        """Retrieve relevant excerpts from the indexed WASDE document for the given query."""
        chunks = vector_search(query, doc_id=doc_id, filters=filters)
        if not chunks:
            return "No relevant excerpts found in the document for this query."
        return "\n\n---\n\n".join(chunks)

    return search_document


def get_agent_client() -> Any:
    """
    Factory to get the language model client for the agent.
    This makes the retrieval agent composable across different model providers
    (e.g., OpenAI, Anthropic) if MAF supports them.
    """
    try:
        from agent_framework.openai import OpenAIChatClient
    except ImportError:
        raise ImportError("Microsoft Agent Framework not installed. Run: pip install agent-framework --pre")
        
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing but required for OpenAIChatClient.")
    
    return OpenAIChatClient(api_key=api_key)


def build_retrieval_agent(doc_id: str, client: Any, filters: dict = None):
    """
    Composable agent builder. Pass in any compatible LLM `client` and filtering strategy.
    """
    from agent_framework import Agent
    
    search_tool = create_search_tool(doc_id, filters=filters)
    
    return Agent(
        client=client,
        instructions=(
            "You are a helpful assistant answering questions about USDA WASDE (World Agricultural Supply and Demand Estimates) reports. "
            "When the user asks a question, first call the search_document tool with their question to retrieve relevant excerpts from the report. "
            "Then answer based ONLY on the retrieved excerpts. Use plain language. If the excerpts do not contain the answer, say so. "
            "Keep answers concise (a few sentences)."
        ),
        tools=[search_tool],
    )


def answer_with_maf(query: str, doc_id: str, filters: dict = None) -> str:
    """
    Use Microsoft Agent Framework (MAF) to answer the query: the agent
    uses a retrieval tool (LanceDB etc.) to get context, then generates an answer.
    """
    try:
        client = get_agent_client()
        agent = build_retrieval_agent(doc_id, client=client, filters=filters)
    except Exception as e:
        return f"Error initializing agent: {e}"

    async def _run():
        result = await agent.run(query)
        return str(result) if result else "No response from the agent."

    try:
        return asyncio.run(_run())
    except Exception as e:
        return "Error from MAF agent: {}.".format(e)
