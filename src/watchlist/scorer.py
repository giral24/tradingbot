"""
Market scoring for watchlist selection.

STRATEGY: Monitor MANY markets for arbitrage opportunities.
- Arbitrage happens when ask_A + ask_B < 1.0
- Can occur in ANY market when:
  1. Market is poorly configured by market maker
  2. Large buy destabilizes prices temporarily
- We need to monitor many markets and react FAST
"""

from dataclasses import dataclass

from src.clob.gamma_client import GammaMarket


@dataclass
class MarketScore:
    """Score breakdown for a market."""

    market: GammaMarket
    total_score: float

    # Component scores (0-1)
    liquidity_score: float  # Can we execute?
    activity_score: float  # Is there trading happening?

    @property
    def condition_id(self) -> str:
        return self.market.condition_id


class MarketScorer:
    """
    Scores markets for monitoring.

    Strategy: Include ANY market where we could execute a trade.
    The real filter is in the fast loop when we detect ask_A + ask_B < 1.0

    Only exclude:
    - Markets with zero liquidity (can't execute)
    - Markets not accepting orders
    """

    # Minimum liquidity to execute trades with decent size
    MIN_LIQUIDITY = 500  # $500 minimum

    def __init__(self, min_liquidity: float = MIN_LIQUIDITY):
        self.min_liquidity = min_liquidity

    def score_market(self, market: GammaMarket) -> MarketScore | None:
        """
        Calculate score for a single market.

        Returns None if market should be excluded.
        """
        # === EXCLUSION FILTERS (minimal) ===

        # Must have minimum liquidity to execute
        if market.liquidity < self.min_liquidity:
            return None

        # === SCORING ===
        # Higher liquidity = easier to execute = higher score
        # But we include all markets, just prioritize by executability

        # Liquidity score (log scale to not over-weight huge markets)
        import math
        liquidity_score = min(math.log10(market.liquidity + 1) / 6, 1.0)  # $1M = 1.0

        # Activity score - markets with activity have price movement
        if market.volume_24h > 0:
            activity_score = min(math.log10(market.volume_24h + 1) / 6, 1.0)
        else:
            activity_score = 0.1  # New markets still interesting

        # Simple average
        total_score = (liquidity_score + activity_score) / 2

        return MarketScore(
            market=market,
            total_score=total_score,
            liquidity_score=liquidity_score,
            activity_score=activity_score,
        )

    def score_markets(self, markets: list[GammaMarket]) -> list[MarketScore]:
        """
        Score multiple markets and return sorted by score (highest first).

        Markets that don't pass filters are excluded.
        """
        scores = []
        for m in markets:
            score = self.score_market(m)
            if score is not None:
                scores.append(score)

        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores
