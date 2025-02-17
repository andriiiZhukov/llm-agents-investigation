import asyncio
from typing import Literal

from llama_index.llms.ollama import Ollama
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core import PromptTemplate
from pydantic import Field

llm = Ollama(model="llama3.1:8b", json_mode=True)


class QueryRoute(BaseModel):
    """Route a user query to the most relevant source."""

    source: Literal["web-search", "llm"] = Field(
        description="Given an user query choose where to route it."
    )


query_router_prompt = """
You are an expert at routing a user query to web-search or llm. Think about the user query, what the user needs.
Choose the correct source which will handle the user query.
When the user needs external real-time knowledge, use web-search.
In all other cases use llm.
User query: {query}
Source:     
"""
query_router_prompt_template = PromptTemplate(query_router_prompt)
query_router = llm.as_structured_llm(QueryRoute)


llm_prompt = """
You are an assistant for question-answering tasks. If the context is given, use it to answer the user query.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Query: {query} 
Context: {context} 
Answer:  
"""
llm_prompt_template = PromptTemplate(llm_prompt)


class WebsearchEvent(Event):
    query: str


class LLMAnswerEvent(Event):
    query: str
    context: str


class FirstWorkflow(Workflow):
    @step
    async def route_query(self, event: StartEvent) -> WebsearchEvent | LLMAnswerEvent | StopEvent:
        query = event.query

        result = query_router.complete(query_router_prompt_template.format(query=query))
        print(f"Route result: {result.model_dump()}")
        source = result.model_dump()["raw"]["source"]
        print(f"Chosen source: {source}")
        if source == "web-search":
            return WebsearchEvent(query=query)
        elif source == "llm":
            return LLMAnswerEvent(query=query, context="")

        return StopEvent(result=result)

    @step
    async def websearch(self, event: WebsearchEvent) -> LLMAnswerEvent:
        query = event.query
        tavily_search_tool = TavilyToolSpec(api_key="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")
        result = tavily_search_tool.search(query=query, max_results=1)
        doc = result[0]
        txt = doc.text
        print(f"Found web context: {txt}")

        return LLMAnswerEvent(query=query, context=txt)

    @step
    async def generate_answer(self, event: LLMAnswerEvent) -> StopEvent:
        query = event.query
        context = event.context

        template = llm_prompt_template.format(query=query, context=context)
        result = llm.complete(template)

        return StopEvent(result=result)


workflow = FirstWorkflow(verbose=True)


#draw_all_possible_flows(FirstWorkflow, filename="basic_workflow.html")

async def run_workflow():
    result = await workflow.run(start_event=StartEvent(query="Who is the president of Poland now?"))
    print(result)


asyncio.run(run_workflow())
