# ==============================================================================
# chatbot/state_manage.py  –  Remember what each user is doing mid-conversation
# ==============================================================================
# WHAT IS THIS?
#   When a user says "add expense" we need to ask them several follow-up
#   questions (how much? what for? what date?). This file remembers where
#   each user is in that multi-step process.
#
# HOW IT WORKS:
#   We keep a dictionary in memory:
#     sessions = {
#       user_id_1: { "flow": "expense", "step": 1, "data": {"amount": 300} },
#       user_id_2: { "flow": None,      "step": 0, "data": {} },
#     }
# ==============================================================================


def _new_session():
    """Return a fresh, empty session."""
    return {
        "flow": None,   # What is the user doing? e.g. "expense", "income", None
        "step": 0,      # Which question are we on? (1, 2, 3 …)
        "data": {},     # Answers collected so far
    }


class StateManager:
    """
    Stores conversation state for every user.

    Usage:
        state = StateManager()
        state.start_flow(user_id, "expense")   # Begin expense entry
        state.save_data(user_id, "amount", 300) # Save amount answer
        state.get_flow(user_id)                 # → "expense"
        state.reset(user_id)                    # Done, clear everything
    """

    def __init__(self):
        # All sessions live here. Key = user_id, Value = session dict.
        self.sessions = {}

    # ── Get or create session ─────────────────────────────────────────────────

    def get(self, user_id):
        """Return the user's session, creating a new one if needed."""
        if user_id not in self.sessions:
            self.sessions[user_id] = _new_session()
        return self.sessions[user_id]

    # ── Start a flow ──────────────────────────────────────────────────────────

    def start_flow(self, user_id, flow_name: str):
        """
        Begin a new multi-step flow (e.g. "expense").
        Resets step to 1 and clears any old data.
        """
        self.sessions[user_id] = {
            "flow": flow_name,
            "step": 1,
            "data": {},
        }
        print(f"[STATE DEBUG] User {user_id}: started flow='{flow_name}'")

    # ── Save a piece of data ──────────────────────────────────────────────────

    def save_data(self, user_id, key: str, value):
        """
        Save one answer inside the session's data dict.
        Example: save_data(user_id, "amount", 300)
        """
        session = self.get(user_id)
        session["data"][key] = value
        print(f"[STATE DEBUG] User {user_id}: saved data['{key}'] = {value}")

    # ── Advance to next step ──────────────────────────────────────────────────

    def next_step(self, user_id):
        """Move to the next question in the flow."""
        session = self.get(user_id)
        session["step"] += 1
        print(f"[STATE DEBUG] User {user_id}: step → {session['step']}")

    # ── Read helpers ──────────────────────────────────────────────────────────

    def get_flow(self, user_id):
        """Return the active flow name, or None if no flow is running."""
        return self.get(user_id)["flow"]

    def get_step(self, user_id):
        """Return the current step number."""
        return self.get(user_id)["step"]

    def get_data(self, user_id):
        """Return all collected data as a dict."""
        return self.get(user_id)["data"]

    def has(self, user_id, key: str) -> bool:
        """Check if a specific key was already collected."""
        return key in self.get(user_id)["data"]

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, user_id):
        """Clear the session completely (flow is done or cancelled)."""
        self.sessions[user_id] = _new_session()
        print(f"[STATE DEBUG] User {user_id}: session reset")