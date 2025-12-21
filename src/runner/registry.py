"""
Bot registry for dynamic bot selection.
"""

from typing import Type

from src.bots.base import BaseBot


class BotRegistry:
    """Registry for available bots."""

    _bots: dict[str, Type[BaseBot]] = {}

    @classmethod
    def register(cls, bot_class: Type[BaseBot]) -> Type[BaseBot]:
        """
        Register a bot class.

        Can be used as a decorator:
            @BotRegistry.register
            class MyBot(BaseBot):
                ...

        Args:
            bot_class: The bot class to register

        Returns:
            The bot class (for decorator usage)
        """
        # Create temporary instance to get the name
        # We need to be careful here - we just need the name property
        name = bot_class.__name__.lower().replace("bot", "")

        # Try to get the actual name from the class
        if hasattr(bot_class, "name") and isinstance(
            getattr(bot_class, "name", None), property
        ):
            # For properties, we need to instantiate or use a different approach
            # Let's use a class attribute instead
            pass

        cls._bots[name] = bot_class
        return bot_class

    @classmethod
    def register_with_name(cls, name: str):
        """
        Register a bot class with a specific name.

        Usage:
            @BotRegistry.register_with_name("my_bot")
            class MyBot(BaseBot):
                ...

        Args:
            name: The name to register the bot under

        Returns:
            Decorator function
        """

        def decorator(bot_class: Type[BaseBot]) -> Type[BaseBot]:
            cls._bots[name] = bot_class
            return bot_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseBot] | None:
        """
        Get a bot class by name.

        Args:
            name: The bot name

        Returns:
            The bot class or None if not found
        """
        return cls._bots.get(name)

    @classmethod
    def list_bots(cls) -> list[str]:
        """
        List all registered bot names.

        Returns:
            List of bot names
        """
        return list(cls._bots.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered bots (mainly for testing)."""
        cls._bots.clear()
