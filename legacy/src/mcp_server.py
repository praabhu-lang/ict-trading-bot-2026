import os
from dotenv import load_dotenv
from tavily import TavilyClient
from fastmcp import FastMCP
from src.execution_engine import TradingExecutionEngine

# Load environment variables
load_dotenv()

# Initialize FastMCP server and services
mcp = FastMCP(name="ICT-Trading-MCP-Server")
engine = TradingExecutionEngine()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def check_market_shock_tool(ticker: str) -> str:
    """
    Queries Tavily for breaking financial news or market shocks for a given ticker.
    """
    try:
        response = tavily_client.search(
            query=f"breaking crash negative news economic shock {ticker}",
            topic="news",
            time_range="day",
            max_results=2
        )
        for res in response.get("results", []):
            content = res.get("content", "").lower()
            if any(kw in content for kw in ["crash", "plunge", "halt", "emergency", "crisis"]):
                return f"[GUARDRAIL WARNING] Market shock detected for {ticker}: {res.get('title')}"
        return f"[GUARDRAIL PASSED] No active market shocks detected for {ticker}."
    except Exception as e:
        return f"[TAVILY ERROR]: {str(e)}. Proceeding with caution."

@mcp.tool()
def execute_credit_spread_tool(long_option_symbol: str, short_option_symbol: str, qty: int, net_limit_price: float) -> str:
    """
    Executes a multi-leg vertical credit spread via Alpaca's MLeg endpoint (symbol=None).
    """
    result = engine.execute_option_spread_trade(
        long_option_symbol=long_option_symbol,
        short_option_symbol=short_option_symbol,
        qty=qty,
        net_limit_price=net_limit_price,
        is_debit=False  # Credit Spread
    ) if hasattr(engine, 'execute_option_spread_trade') else None
    
    if result:
        return f"Successfully queued MLeg credit spread order ID: {result.id} | Status: {result.status}"
    return "Credit spread order execution failed."

@mcp.tool()
def monitor_and_manage_positions_tool() -> str:
    """
    Monitors active positions and enforces client-side stop-losses (-150%) and profit targets (+50%).
    """
    try:
        engine.monitor_and_manage_positions(stop_loss_pct=-150.0, profit_target_pct=50.0)
        return "Position monitoring and risk management cycle completed successfully."
    except Exception as e:
        return f"Error managing positions: {str(e)}"

if __name__ == "__main__":
    # Run over HTTP transport on port 8000
    mcp.run(transport="http", host="127.0.0.1", port=8000)