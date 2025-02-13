from pprint import pprint
from typing import TypedDict, Literal
import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.graph import END, StateGraph, START
from langchain_community.tools import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver

embeddings = OllamaEmbeddings(model="llama3.1:8b")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="collection_name",
    path="path_to_vectorstore"
)

llm = ChatOllama(model="llama3.1:8b", temperature=0)


class GraphState(TypedDict):
    query: str
    final_answer: AIMessage
    documents: list[Document]
    answer_score: str
    query_route_name: str
    rewrite_query_counter: int


class QueryRoute(BaseModel):
    """Route a user query to the most relevant source."""

    source: Literal["vectorstore", "web-search", "llm"] = Field(
        description="Given an user query choose where to route it."
    )


query_router_prompt = """
You are an expert at routing a user query to a vectorstore or web-search or llm.
The vectorstore contains documents related to retrieval-augmented generation and large language models.
Use the vectorstore for queries on these topics. Otherwise, when the external real-time knowledge is needed, use web-search.
In all other cases use llm.
Query: {query}
Source:     
"""
query_router_prompt_template = ChatPromptTemplate.from_template(query_router_prompt)
query_router = query_router_prompt_template | llm.with_structured_output(QueryRoute)


def run_query_router(state: GraphState):
    query = state["query"]
    query_route = query_router.invoke({"query": query})
    return {"query_route_name": query_route.source}


def check_query_route(state: GraphState):
    query_route = state["query_route_name"]
    if query_route == "web-search":
        print("Route to web search")
        return "web_search"
    elif query_route == "vectorstore":
        print("Route to vectorstore")
        return "vectorstore"
    return "llm_answer"


class DocumentAnswer(BaseModel):
    """Number of documents which are relevant to the user query"""

    relevant_document_number: int = Field(
        description="Number of relevant document"
    )


evaluate_retrieved_docs_prompt = """
You are a grader assessing whether retrieved documents from the vector store are relevant to the user query.
There will be ten documents given. They are numbered from 1 to 10.
You must assess each of them with the given user query and return a number of document which you think is relevant.
If you won't find any relevant document you must return 0.
User query: {query}
Documents: {docs}
Relevant document number: 
"""
retrieved_docs_evaluator_prompt_template = ChatPromptTemplate.from_template(evaluate_retrieved_docs_prompt)
retrieved_docs_evaluator = retrieved_docs_evaluator_prompt_template | llm.with_structured_output(DocumentAnswer)


def search_and_evaluate_docs(state: GraphState):
    query = state["query"]
    docs_filter = Filter(must=[FieldCondition(key="metadata.category", match=MatchValue(value="NarrativeText"))])
    retrieved_docs = vector_store.similarity_search(query, k=10, filter=docs_filter)
    i = 1
    docs_txt = []
    for d in retrieved_docs:
        txt = str(i) + ". " + d.page_content
        i = i + 1
        docs_txt.append(txt)
    documents_txt = "\n".join(docs_txt)

    result = retrieved_docs_evaluator.invoke({"query": query, "docs": documents_txt})
    relevant_document_number = result.model_dump()["relevant_document_number"]
    if relevant_document_number > 0:
        print("Relevant doc is found")
        relevant_doc = retrieved_docs[relevant_document_number - 1]
        return {"documents": [relevant_doc]}
    else:
        print("Relevant doc is not found")
        return {"documents": []}


web_search_tool = TavilySearchResults()


def web_search(state: GraphState):
    query = state["query"]

    docs = web_search_tool.invoke({"query": query})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": [web_results]}


rag_prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the user query.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Query: {query} 
Context: {context} 
Answer:
"""
rag_template = ChatPromptTemplate.from_template(rag_prompt)
rag_agent = rag_template | llm


def generate_answer(state: GraphState):
    query = state["query"]
    docs = state["documents"]
    context = ""
    if docs:
        context = docs[0].page_content

    result = rag_agent.invoke({"query": query, "context": context})
    return {"final_answer": result}


def llm_answer(state: GraphState):
    query = state["query"]
    result = llm.invoke(query)
    return {"final_answer": result}


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


answer_grade_prompt = """
You are a grader assessing whether an answer addresses / resolves an user query.
Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the query.
Query: {query} 
Answer: {answer} 
Score:
"""
answer_grade_template = ChatPromptTemplate.from_template(answer_grade_prompt)
structured_llm_grader = llm.with_structured_output(GradeAnswer)
answer_grader_agent = answer_grade_template | structured_llm_grader


def grade_answer(state: GraphState):
    query = state["query"]
    answer = state["final_answer"].content

    result = answer_grader_agent.invoke({"query": query, "answer": answer})
    score = result.model_dump()["binary_score"]

    return {"answer_score": score}


def check_answer_grade(state: GraphState):
    try:
        if state["rewrite_query_counter"] == 1:
            return "end"
    except KeyError:
        pass

    score = state["answer_score"]
    if score == 'no':
        return "rewrite_query"
    else:
        return "answer is ok"


class UpdatedQuery(BaseModel):
    query: str = Field(
        description="Defines a new improved query"
    )


query_rewrite_template = """
You are a query re-writer that converts an input query to a better version that is optimized for vectorstore retrieval.
Look at the input query and try to reason about the underlying semantic intent / meaning.
Generate an improved query.
Query: {query}
Improved query:     
"""
query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)
query_rewriter = query_rewrite_prompt | llm.with_structured_output(UpdatedQuery)


def rewrite_query(state: GraphState):
    query = state["query"]
    result = query_rewriter.invoke({"query": query})
    new_query = result.model_dump()["query"]
    print(f"Updated query: {new_query}")
    return {"query": new_query, "rewrite_query_counter": 1}


workflow = StateGraph(GraphState)

workflow.add_node("route_query", run_query_router)
workflow.add_node("vectorstore", search_and_evaluate_docs)
workflow.add_node("web_search", web_search)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("grade_answer", grade_answer)
workflow.add_node("rewrite_query", rewrite_query)

workflow.add_edge(START, "route_query")
workflow.add_conditional_edges("route_query", check_query_route,
                               {"vectorstore": "vectorstore", "web_search": "web_search",
                                "llm_answer": "llm_answer"})

workflow.add_edge("vectorstore", "generate_answer")
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("llm_answer", END)

workflow.add_edge("generate_answer", "grade_answer")

workflow.add_conditional_edges("grade_answer", check_answer_grade,
                               {"rewrite_query": "rewrite_query", "answer is ok": END, "end": END})

workflow.add_edge("rewrite_query", "route_query")

in_memory_checkpoint_saver = MemorySaver()
app = workflow.compile(checkpointer=in_memory_checkpoint_saver)


async def run_app(query: str, thread_id: str):
    query = {"query": query}
    config = {"configurable": {"thread_id": thread_id}}
    async for output in app.astream(query, config):
        for key, value in output.items():
            pprint(f"Node '{key}', thread_id={thread_id}")
            # pprint(f"'{value}'")
            if "final_answer" in value:
                pprint(f"'Final answer: {value["final_answer"].content}'")
            if "answer_score" in value:
                pprint(f"'Answer score: {value["answer_score"]}'")
        print("-------------------------------------------------------------------------")


async def execute_queries():
    await asyncio.gather(
        run_app("Hello!", "1"),
        run_app("What's up?", "2"),
        run_app("What issues LLMs are struggling?", "3"),
        run_app("Who is the current prime minister in Poland?", "4"),
    )


asyncio.run(execute_queries())
