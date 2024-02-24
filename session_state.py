class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def get_session_state(session_id, **kwargs):
    if session_id not in session_states:
        session_states[session_id] = SessionState(**kwargs)
    return session_states[session_id]

session_states = {}
