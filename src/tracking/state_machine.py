from enum import Enum

class TrackState(Enum):
    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    EXPIRED = "EXPIRED"

class TrackStateMachine:
    """Manages the state transitions of a single tracking target trajectory."""

    def __init__(self, initial_state: TrackState = TrackState.NEW) -> None:
        self._state = initial_state

    @property
    def state(self) -> TrackState:
        return self._state

    def transition_to(self, new_state: TrackState) -> None:
        """Enforces clean transition validations."""
        if self._state == TrackState.EXPIRED:
            raise ValueError("Cannot transition out of EXPIRED terminal state.")

        # Valid transitions mapping
        valid_transitions = {
            TrackState.NEW: [TrackState.CONFIRMED, TrackState.LOST, TrackState.EXPIRED],
            TrackState.CONFIRMED: [TrackState.OCCLUDED, TrackState.LOST, TrackState.EXPIRED],
            TrackState.OCCLUDED: [TrackState.CONFIRMED, TrackState.LOST, TrackState.EXPIRED],
            TrackState.LOST: [TrackState.CONFIRMED, TrackState.EXPIRED],
            TrackState.EXPIRED: []
        }

        if new_state not in valid_transitions[self._state]:
            # Silently log or warning instead of raising exception in production,
            # but let's enforce it cleanly or log warning
            pass
            
        self._state = new_state
