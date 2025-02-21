from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgres://user:password@localhost:5432/langgraph"


class GraphState(TypedDict):
    query: str
    final_answer: AIMessage
    documents: list[Document]


def node_a(state: GraphState):
    print("node A")
    message = AIMessage(content="message from node A")
    doc = Document(page_content="doc content")
    query = state["query"]
    return {"query": query, "final_answer": message, "documents": [doc]}

def node_b(state: GraphState):
    print("node B")
    message = AIMessage(content="message from node B")
    return {"final_answer": message}


workflow = StateGraph(GraphState)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["node_b"])

    config = {"configurable": {"thread_id": "1"}}
    initial_input = {"query": "What's up?"}

    graph.invoke(initial_input, config)

    state = graph.get_state(config)
    print(state)

    try:
        user_approval = input("Do you want to go to the next step? (yes/no): ")
    except:
        user_approval = "yes"

    if user_approval.lower() == "yes":
        graph.invoke(None, config)
    else:
        print("Operation cancelled by user.")

    state = graph.get_state(config)
    print(state)



