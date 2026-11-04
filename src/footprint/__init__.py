"""Order-flow (footprint) engine: our own bid/ask volume per price, built from SIP trade ticks.

Modules
  ticks       fetch + cache SIP trades, classify the aggressor, store 1-minute footprints
  core        FootprintBar / FootprintBuilder, delta and 400% diagonal imbalances
  regime      GEXRegimeFilter (gamma flip, walls)
  location    VolumeProfileLocationEngine (FRVP, VAH/VAL/POC, swings, Fibonacci pockets)
  absorption  FootprintAbsorptionDetector (absorption -> dominance-shift state machine)
  execution   ExecutionManager (entry styles, wick stop, wall targets, trailing)
  live        websocket feed -> the same builder, signals only (never sends orders)

Backtest: python -m src.backtest_footprint
"""
