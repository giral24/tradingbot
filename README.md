# Polymarket Trading Bot

Automated trading bot for [Polymarket](https://polymarket.com) prediction markets.

## Strategies

### 1. Intra-market Arbitrage
Detects arbitrage opportunities when `ask_A + ask_B < 1.0` and executes both sides to lock in risk-free profit.

```
Example:
  ask_YES = 0.48, ask_NO = 0.51
  Total cost = 0.99
  Guaranteed payout = 1.00
  Profit = 1% (risk-free)
```

### 2. Mean Reversion
Detects sudden price spikes (≥8% in <2 min) caused by large orders and trades the opposite direction, expecting price to revert.

```
Example:
  Price drops: 0.50 → 0.42 (large sell)
  Bot buys at: 0.42, 0.40, 0.38 (scaled entry)
  Price reverts: 0.45
  Profit = ~7%
```

## Features

- Real-time orderbook monitoring via WebSocket
- Automatic market discovery and filtering
- Risk management with position limits
- Dry-run mode for paper trading
- Metrics collection and logging
- Configurable parameters

## Installation

```bash
# Clone repository
git clone https://github.com/giral24/polymarket-bot.git
cd polymarket-bot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env.local
```

Edit `.env.local` with your settings:

```env
# For live trading (optional)
PRIVATE_KEY=your_polygon_wallet_private_key

# Bot settings
DRY_RUN=true
TRADE_SIZE=1.0
```

## Usage

### Arbitrage Bot

```bash
# Dry run (no real trades)
python scripts/run_bot.py --dry-run

# With verbose logging
python scripts/run_bot.py --dry-run -v

# Live trading
python scripts/run_bot.py --live --trade-size 10.0
```

### Mean Reversion Bot

```bash
# Dry run
python scripts/run_mean_reversion.py --dry-run

# Custom parameters
python scripts/run_mean_reversion.py --dry-run \
  --trade-size 10.0 \
  --threshold 0.08 \
  --min-liquidity 1000 \
  --max-liquidity 50000
```

### Run in Background

```bash
# Arbitrage bot
nohup python scripts/run_bot.py --dry-run > bot_arb.log 2>&1 &

# Mean reversion bot
nohup python scripts/run_mean_reversion.py --dry-run > bot_mr.log 2>&1 &

# Check logs
tail -f bot_arb.log
grep -i arbitrage bot_arb.log

# Stop bots
pkill -f run_bot.py
pkill -f run_mean_reversion.py
```

## Project Structure

```
polymarket-bot/
├── src/
│   ├── bots/
│   │   ├── arb_intramarket/   # Arbitrage strategy
│   │   └── mean_reversion/    # Mean reversion strategy
│   ├── clob/                  # Polymarket API clients
│   ├── ws/                    # WebSocket client
│   ├── watchlist/             # Market selection
│   ├── risk/                  # Risk management
│   └── metrics/               # Metrics collection
├── scripts/                   # Execution scripts
├── tests/                     # Unit tests
└── data/                      # Runtime data (gitignored)
```

## Parameters

### Arbitrage Bot
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trade-size` | 1.0 | Trade size in USD |
| `--min-spread` | 0.001 | Minimum spread (0.1%) |
| `--dry-run` | true | Paper trading mode |

### Mean Reversion Bot
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trade-size` | 10.0 | Size per entry (max 3 entries) |
| `--threshold` | 0.08 | Price change trigger (8%) |
| `--min-liquidity` | 1000 | Min market liquidity |
| `--max-liquidity` | 50000 | Max market liquidity |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_mean_reversion.py -v
```

## Disclaimer

This software is for educational purposes only. Trading on prediction markets involves risk. Use at your own risk. The authors are not responsible for any financial losses.

## License

MIT
