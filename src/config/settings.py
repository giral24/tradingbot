"""
Central configuration system using Pydantic.
All settings loaded from environment variables.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Bot Selection
    bot_name: str = Field(default="arb_intramarket", description="Bot to run")
    dry_run: bool = Field(default=True, description="Paper trading mode")

    # Wallet Configuration (API credentials derived automatically)
    private_key: str = Field(default="", description="Wallet private key")
    polymarket_proxy_address: str = Field(
        default="",
        description="Polymarket proxy wallet address (shown under profile picture on polymarket.com)",
    )

    # API Configuration
    clob_api_url: str = Field(
        default="https://clob.polymarket.com",
        description="CLOB REST API URL",
    )
    gamma_api_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Gamma API URL for market discovery",
    )
    clob_ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        description="CLOB WebSocket URL",
    )
    chain_id: int = Field(default=137, description="Chain ID (137 = Polygon)")

    # Watchlist Configuration
    watchlist_refresh_interval: int = Field(
        default=300,
        description="Watchlist refresh interval in seconds",
    )
    watchlist_max_size: int = Field(
        default=50,
        description="Maximum markets in watchlist",
    )
    watchlist_entry_threshold: float = Field(
        default=0.7,
        description="Score threshold to enter watchlist",
    )
    watchlist_exit_threshold: float = Field(
        default=0.3,
        description="Score threshold to exit watchlist",
    )

    # Arbitrage Configuration
    arb_min_edge: float = Field(
        default=0.005,
        description="Minimum edge for arbitrage (0.005 = 0.5%)",
    )
    arb_min_liquidity: float = Field(
        default=10.0,
        description="Minimum liquidity in USD",
    )
    arb_max_position: float = Field(
        default=100.0,
        description="Maximum position size per trade in USD",
    )

    # Risk Management
    risk_max_exposure: float = Field(
        default=1000.0,
        description="Maximum total exposure in USD",
    )
    risk_max_open_orders: int = Field(
        default=10,
        description="Maximum concurrent open orders",
    )
    risk_kill_switch_loss: float = Field(
        default=500.0,
        description="Kill switch loss threshold in USD",
    )
    risk_leg_timeout: float = Field(
        default=5.0,
        description="Timeout for second leg in seconds",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )

    # Storage
    db_path: Path = Field(
        default=Path("data/bot.db"),
        description="SQLite database path",
    )
    watchlist_path: Path = Field(
        default=Path("data/watchlist.json"),
        description="Watchlist JSON path",
    )

    def has_credentials(self) -> bool:
        """Check if wallet private key is configured."""
        return bool(self.private_key)

    def ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.watchlist_path.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
