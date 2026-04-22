from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from tools.market_tools import get_stock_price
from state import TradeState

llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
price_llm = llm.bind_tools([get_stock_price])

def price_agent(state: TradeState) -> TradeState:
    """Node: fetches current price and writes it to state."""
    response = price_llm.invoke([
        HumanMessage(content=f"Get the current stock price for {state['ticker']}.")
    ])
    # Extract the tool result from the response
    for block in response.tool_calls:
        if block["name"] == "get_stock_price":
            result = get_stock_price.invoke(block["args"])
            state["current_price"] = result["price"]
    return state