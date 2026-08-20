from dotenv import load_dotenv
load_dotenv()

from graph.workflow import graph
from state import TradeState

def run(ticker: str):
    initial_state: TradeState = {
        "ticker": ticker.upper(),
        "current_price": None,
        "headlines": None,
        "rsi": None,
        "confidence": None,
        "report": None,
        "needs_human_review": False,
    }
    result = graph.invoke(initial_state)
    print("\n── NexusTrade Report ──────────────────────────")
    print(result["report"])
    print(f"\nConfidence: {result['confidence']}")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run(ticker)