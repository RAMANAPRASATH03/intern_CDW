from langchain_groq import ChatGroq
def create_chat_groq():
    """
    Initializes and returns the Groq LLM.
    Returns:
        ChatGroq: The Groq LLM instance.
    """
    return ChatGroq(
        model="mixtral-8x7b-32768",  # Use any supported model
        temperature=0,  # Set temperature for deterministic output
        max_tokens=None,  # No limit on the number of tokens
        timeout=None,  # No timeout
        max_retries=2  # Retry up to 2 times in case of failure
    )