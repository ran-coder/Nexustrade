# agents/report_generator.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from state import TradeState
import re

llm = ChatGroq(temperature=0, model='llama-3.3-70b-versatile')

def report_generator(state: TradeState) -> TradeState:
    prompt = f"""
    You are a stock research analyst. Based on the data below, write a
    2-paragraph investment summary for {state['ticker']} and end with
    a confidence score between 0.0 and 1.0 on a new line like: CONFIDENCE: 0.82

    Current Price: ${state['current_price']}
    RSI: {state['rsi']}
    Recent Headlines:
    {chr(10).join(f"- {h}" for h in state['headlines'])}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content

    # Parse confidence from the response
    match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
    confidence = float(match.group(1)) if match else 0.5

    state["report"] = text
    state["confidence"] = confidence
    state["needs_human_review"] = confidence < 0.6
    return state