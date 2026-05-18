from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tools.market_tools import get_news_headlines
from state import TradeState

llm = ChatGroq(model="llama-3.3-70b-versatile")
news_llm = llm.bind_tools([get_news_headlines])

def news_agent(state: TradeState) -> TradeState:
    """Node: fetches current news and writes it to state."""
    response = news_llm.invoke([
        HumanMessage(content=f"Get the current stock price news for {state['ticker']}.")
    ])
    # Extract the tool result from the response
    for block in response.tool_calls:
        if block["name"] == "get_news_headlines":
            result = get_news_headlines.invoke(block["args"])
            state["current_news"] = result["title"]
    return state