from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tools.market_tools import get_rsi
from state import TradeState

llm = ChatGroq(model="llama-3.3-70b-versatile")
technical_llm = llm.bind_tools([get_rsi])

def technical_agent(state: TradeState) -> TradeState:
    """Node: fetches current rsi and writes it to state."""
    response = technical_llm.invoke([
        HumanMessage(content=f"Get the current stock price for {state['ticker']}.")
    ])
    # Extract the tool result from the response
    for block in response.tool_calls:
        if block["name"] == "get_rsi":
            result = get_rsi.invoke(block["args"])
            state["current_rsi"] = result["rsi"]
    return state