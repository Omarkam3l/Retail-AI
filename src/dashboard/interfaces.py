from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDashboardRenderer(ABC):
    """Abstract interface defining the contract for dashboard page renderings."""

    @abstractmethod
    def render_analytics(self) -> Dict[str, Any]:
        """Fetches and prepares metadata tables for rendering analytics dashboards.

        Returns:
            A dictionary of compiled visual dashboard widgets.
        """
        pass
