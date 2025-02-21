from typing import TypedDict, List

from langchain_core.messages import AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.store.postgres import PostgresStore
import random

DB_URI = "postgres://user:password@localhost:5432/langgraph"


class GraphState(TypedDict):
    messages_history: List[str]
    current_message: str


def node_a(state: GraphState):

    random_num = random.randint(1, 1000)
    message = AIMessage(content=str(random_num))

    messages_history = []
    if "messages_history" in state:
        messages_history.extend(state["messages_history"])
    messages_history.append(message.content)

    return {"current_message": message.content, "messages_history": messages_history}


workflow = StateGraph(GraphState)
workflow.add_node(node_a)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", END)

app = workflow.compile()


with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()

    # Define namespace
    user_id = "123"
    application = "prompt repo"
    namespace = (user_id, application)

    #First run
    store_item = store.get(namespace, "messages")
    messages_history = store_item.value["values"] if store_item else []

    initial_input = {"messages_history": []}
    result = app.invoke(initial_input)
    store.put(namespace, "messages", {"values": result["messages_history"]})
    print(result["messages_history"])


    #Second run
    store_item = store.get(namespace, "messages")
    messages_history = store_item.value["values"]

    initial_input = {"messages_history": messages_history}
    result = app.invoke(initial_input)
    store.put(namespace, "messages", {"values": result["messages_history"]})
    print(result["messages_history"])


    #Third run
    store_item = store.get(namespace, "messages")
    messages_history = store_item.value["values"]

    initial_input = {"messages_history": messages_history}
    result = app.invoke(initial_input)
    store.put(namespace, "messages", {"values": result["messages_history"]})
    print(result["messages_history"])
