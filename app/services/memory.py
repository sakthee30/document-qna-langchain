chat_memory = {}

def get_session_history(session_id: str):
    if session_id not in chat_memory:
        chat_memory[session_id] = []
    return chat_memory[session_id]

def add_to_history(session_id: str, user_msg: str, ai_msg: str):
    chat_memory[session_id].append(f"User: {user_msg}")
    chat_memory[session_id].append(f"AI: {ai_msg}")