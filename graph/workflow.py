from langgraph.graph import StateGraph, END
from state import TradeState
from agents.price_agent import price_agent
from agents.news_agent import news_agent
from agents.technical_agent import technical_agent
from agents.report_generator import report_generator

def human_review_node(state: TradeState) -> TradeState:
    """Pause and ask the user to confirm before proceeding."""
    print("\n⚠  Low confidence signal. Human review required.")
    print(f"   Ticker: {state['ticker']}")
    print(f"   Confidence: {state['confidence']}")
    approval = input("   Approve this report? (yes/no): ").strip().lower()
    if approval != "yes":
        state["report"] = "Report rejected by human reviewer."
    return state

def should_review(state: TradeState) -> str:
    """Conditional edge: route based on confidence threshold."""
    return "human_review" if state["needs_human_review"] else "end"

# Build the graph
builder = StateGraph(TradeState)

# Add nodes
builder.add_node("price_agent", price_agent)
builder.add_node("news_agent", news_agent)
builder.add_node("technical_agent", technical_agent)
builder.add_node("report_generator", report_generator)
builder.add_node("human_review", human_review_node)

# Set entry point
builder.set_entry_point("price_agent")

# Sequential edges (price → news → technical → report)
builder.add_edge("price_agent", "news_agent")
builder.add_edge("news_agent", "technical_agent")
builder.add_edge("technical_agent", "report_generator")

# Conditional edge after report generation
builder.add_conditional_edges(
    "report_generator",
    should_review,
    {"human_review": "human_review", "end": END}
)
builder.add_edge("human_review", END)

# Compile
graph = builder.compile()