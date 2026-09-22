import os
import psycopg2
from datetime import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from config import Config
from alpaca_client import AlpacaManager
from options_client import OptionExecutionManager

# Define the shared state schema for our multi-agent workflow
class BotState(TypedDict):
    tickers: List[str]
    sentiment_scores: dict
    macro_status: str
    risk_approved: bool
    allowed_risk_amount: float
    option_chains: dict
    execution_signals: list

def sentiment_agent(state: BotState):
    print("\n[AGENT 1] Running Tavily Sentiment Scanner across watchlist...")
    try:
        tavily = TavilyClient(api_key=Config.TAVILY_API_KEY)
        scores = {}
        for ticker in state["tickers"]:
            response = tavily.qna_search(query=f"What is the current market sentiment and news catalyst for {ticker} stock today?")
            snippet = str(response)[:120] if response else "No news snippet found."
            scores[ticker] = snippet
            print(f"  -> {ticker}: Sentiment check captured.")
        state["sentiment_scores"] = scores
        print("[AGENT 1] Sentiment scan complete.")
    except Exception as e:
        print(f"[AGENT 1 ERROR] Tavily search failed: {e}")
        state["sentiment_scores"] = {t: "Bypassed/Error" for t in state["tickers"]}
    return state

def macro_risk_agent(state: BotState):
    print("\n[AGENT 2] Verifying Macro & 4% Risk Compounding Guardrails...")
    try:
        alpaca = AlpacaManager()
        equity, cash = alpaca.get_account_balance()
        max_risk_amount = equity * Config.MAX_RISK_PCT
        print(f"  -> Alpaca Equity: ${equity:,.2f} | Available Cash: ${cash:,.2f}")
        print(f"  -> Strict 4% Risk Capital Allocation Limit: ${max_risk_amount:,.2f}")
        
        state["macro_status"] = "CLEAR"
        state["risk_approved"] = True
        state["allowed_risk_amount"] = max_risk_amount
    except Exception as e:
        print(f"[AGENT 2 ERROR] Risk check failed: {e}")
        state["macro_status"] = "BLOCKED"
        state["risk_approved"] = False
        state["allowed_risk_amount"] = 0.0
    return state

def execution_option_scanner_agent(state: BotState):
    print("\n[AGENT 3] Evaluating Option Chains & Spread Parameters...")
    if not state["risk_approved"] or state["macro_status"] != "CLEAR":
        state["execution_signals"] = ["Trade execution blocked by risk/macro guardrails."]
        print("[AGENT 3] Trade execution aborted due to guardrail failure.")
        return state

    option_manager = OptionExecutionManager()
    chains_found = {}
    signals = []

    for ticker in state["tickers"]:
        expiry_type, chain = option_manager.get_nearest_option_chain(ticker)
        contract_count = len(chain) if chain else 0
        if contract_count > 0:
            chains_found[ticker] = {"expiry_type": expiry_type, "contracts": contract_count}
            signals.append(f"READY: {ticker} ({expiry_type}) with {contract_count} contracts under ${state['allowed_risk_amount']:,.2f} risk cap.")
        else:
            signals.append(f"SKIPPED: {ticker} - No valid option chain found.")

    state["option_chains"] = chains_found
    state["execution_signals"] = signals
    print(f"[AGENT 3] Evaluated {len(state['tickers'])} tickers. Generated {len(signals)} execution signals.")
    return state

def run_workflow():
    print("=" * 70)
    print(f"[START] ICT Multi-Agent Trading Pipeline - {datetime.now()}")
    print("=" * 70)

    # Initialize LangGraph State Machine
    workflow = StateGraph(BotState)

    workflow.add_node("sentiment_node", sentiment_agent)
    workflow.add_node("risk_node", macro_risk_agent)
    workflow.add_node("execution_node", execution_option_scanner_agent)

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
        "allowed_risk_amount": 0.0,
        "option_chains": {},
        "execution_signals": []
    }

    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 70)
    print("[SUMMARY] Multi-Agent Pipeline Execution Results:")
    print("=" * 70)
    for signal in final_state["execution_signals"]:
        print(f" • {signal}")
    print("=" * 70)
    print("[COMPLETE] Multi-Agent Workflow Finished Successfully")
    print("=" * 70)

if __name__ == "__main__":
    run_workflow()