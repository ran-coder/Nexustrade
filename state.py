from typing import TypedDict, Optional, List

class TradeState(TypedDict):
    ticker: str                       # Input: e.g. "AAPL"
    current_price: Optional[float]    # Filled by PriceAgent
    headlines: Optional[List[str]]    # Filled by NewsAgent
    rsi: Optional[float]              # Filled by TechnicalAgent
    confidence: Optional[float]       # Filled by ReportGenerator
    report: Optional[str]             # Final output
    needs_human_review: bool          # Router flag