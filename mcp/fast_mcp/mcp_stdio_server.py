from mcp.server.fastmcp import FastMCP
from langgraph.store.postgres import PostgresStore
import random


DB_URI = "postgres://user:password@localhost:5432/database"


mcp = FastMCP("mcp_server")


@mcp.tool()
def generate_random_number(range_min: int, range_max: int) -> int:
    """Generate random number from given range"""
    random_num = random.randint(range_min, range_max)
    return random_num


@mcp.tool()
def store_number(number: int) -> None:
    """Store number in the database"""
    with PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()

        user_id = "456"
        application = "mcp client"
        namespace = (user_id, application)

        existing_numbers = store.get(namespace, "user_numbers")
        if existing_numbers:
            numbers = existing_numbers.value["numbers"]
            numbers.append(number)
        else:
            numbers = [number]

        store.put(namespace, "user_numbers", {"numbers": numbers})


if __name__ == "__main__":
    mcp.run()
