# NexusTrade 📈

**A multi-agent stock market research assistant powered by LangGraph and Groq.**

NexusTrade orchestrates a team of specialized AI agents that work together to research a stock ticker, analyze price action, gather relevant news, and produce a concise research report with a confidence score — flagging cases that need human review.

---

## How It Works

NexusTrade uses [LangGraph](https://www.langchain.com/langgraph) to coordinate multiple agents, each responsible for a specific piece of the research pipeline. Given a stock ticker, the system:

1. **Fetches the current price** of the stock
2. **Gathers recent news headlines** related to the company
3. **Runs technical analysis** (e.g. RSI) on the stock
4. **Synthesizes findings** into a final research report
5. **Flags low-confidence results** for human review

The agents pass a shared state object between each other as they work, building up the final report step by step.

## Usage

Run the assistant from the command line with a stock ticker:

```bash
python main.py AAPL
```

If no ticker is provided, it defaults to `AAPL`:

```bash
python main.py
```

### Example Output

```
── NexusTrade Report ──────────────────────────
[Generated research report summarizing price action, news, and technical signals]

Confidence: 0.82
```

---

## State

Each run tracks a shared `TradeState` as it moves through the graph:

| Field                | Description                                      |
|-----------------------|---------------------------------------------------|
| `ticker`              | Stock symbol being analyzed                       |
| `current_price`       | Latest fetched price                              |
| `headlines`           | Recent news headlines related to the ticker       |
| `rsi`                 | Computed technical indicator (RSI)                |
| `confidence`          | Model's confidence in the generated report        |
| `report`              | Final synthesized research report                 |
| `needs_human_review`  | Flag indicating low-confidence results            |

---

## Tech Stack

- [LangGraph](https://www.langchain.com/langgraph) — agent orchestration
- [LangChain](https://www.langchain.com/) — LLM tooling and agent framework
- [Groq](https://groq.com/) — fast LLM inference (`llama-3.3-70b-versatile`)
- [LangSmith](https://smith.langchain.com/) — tracing and observability (optional)

---

## Disclaimer

NexusTrade is a research and educational tool. It does not constitute financial advice. Always do your own due diligence before making investment decisions.
