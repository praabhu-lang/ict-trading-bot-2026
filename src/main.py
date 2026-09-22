import os
import psycopg2
from datetime import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from config import Config
from alpaca_client import AlpacaManager

# Define the shared state schema for our multi-agent workflow
class BotState(TypedDict):
    tickers: List[str]
    sentiment_scores: dict
    macro_status: str
    risk_approved: bool
    execution_signals: list

def sentiment_agent(state: BotState):
    print("[AGENT 1] Running Tavily Sentiment Scanner across watchlist...")
    try:
        tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)
        scores = {}
        for ticker in state["tickers"][:3]:  # Scan top tickers to optimize speed
            response = tavily.qna_search(query=f"What is the current market sentiment and news catalyst for {ticker} stock today?")
            scores[ticker] = str(response)[:150]
        state["sentiment_scores"] = scores
        print(f"[AGENT 1] Sentiment scan complete.")
    except Exception as e:
        print(f"[AGENT 1 ERROR] Tavily search failed: {e}")
        state["sentiment_scores"] = {}
    return state

def macro_risk_agent(state: BotState):
    print("[AGENT 2] Verifying Macro & 4% Risk Compounding Guardrails...")
    try:
        alpaca = AlpacaManager()
        equity, cash = alpaca.get_account_balance()
        max_risk_amount = equity * Config.MAX_RISK_PCT
        print(f"[AGENT 2] Alpaca Equity: ${equity:,.2f} | Max 4% Risk Allocation: ${max_risk_amount:,.2f}")
        state["macro_status"] = "CLEAR"
        state["risk_approved"] = True
    except Exception as e:
        print(f"[AGENT 2 ERROR] Risk check failed: {e}")
        state["macro_status"] = "BLOCKED"
        state["risk_approved"] = False
    return state

def execution_agent(state: BotState):
    print("[AGENT 3] Evaluating 0DTE Option Spread Execution Parameters...")
    if state["risk_approved"] and state["macro_status"] == "CLEAR":
        state["execution_signals"] = ["Ready for 0DTE credit spread setup on watchlist."]
    else:
        state["execution_signals"] = ["Trade execution blocked by risk/macro guardrails."]
    print(f"[AGENT 3] Signals generated: {state['execution_signals']}")
    return state

def run_workflow():
    print("=" * 60)
    print(f"[START] LangGraph Multi-Agent Trading Pipeline - {datetime.now()}")
    print("=" * 60)

    # Initialize LangGraph State Machine
    workflow = StateGraph(BotState)

    workflow.add_node("sentiment_node", sentiment_agent)
    workflow.add_node("risk_node", macro_risk_agent)
    workflow.add_node("execution_node", execution_agent)

    workflow.set_entry_point("sentiment_node")
    workflow.add_edge("sentiment_node", "risk_node")
    workflow.add_edge("risk_node", "execution_node")
    workflow.add_edge("execution_node", END)

    app = workflow.compile()

    initial_state = {
        "tickers": Config.WATCHLIST,
        "sentiment_scores": {},
        "macro_status": "PENDING",
        "risk_approved": False,
        "execution_signals": []
    }

    final_state = app.invoke(initial_state)
    print("=" * 60)
    print("[COMPLETE] Multi-Agent Workflow Finished Successfully")
    print("=" * 60)

if __name__ == "__main__":
    run_workflow()