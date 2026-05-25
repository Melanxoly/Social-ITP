from __future__ import annotations

from typing import Optional, Protocol

from social_itp.schemas.types import Action, Observation


class Persuader(Protocol):
    def propose(
        self,
        obs: Observation,
        reply_to_node_id: str,
        reply_to_user_id: Optional[str] = None,
        author_user_id: str = "persuader",
        author_user_name: str = "persuader_bot",
    ) -> Action: ...
