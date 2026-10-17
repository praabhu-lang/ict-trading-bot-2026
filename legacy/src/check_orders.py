from src.execution_engine import TradingExecutionEngine

def check_pending_orders():
    print("==================================================")
    print("[INSPECTION] Checking Alpaca Open/Pending Orders...")
    print("==================================================")
    
    engine = TradingExecutionEngine()
    orders = engine.get_open_orders()
    
    if not orders:
        print(" -> No pending or open orders found.")
    else:
        print(f" -> Found {len(orders)} open/pending order(s):")
        for o in orders:
            print(f"    - ID: {o.id}")
            print(f"      Status: {o.status}")
            print(f"      Type: {o.type} | Side: {o.side}")
            print(f"      Submitted At: {o.submitted_at}")
            print("-" * 40)
            
    print("==================================================")

if __name__ == "__main__":
    check_pending_orders()
