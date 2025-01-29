from model import create_chat_groq
from prompt import code_generator_prompt_from_hub,code_generator_prompt


def generate_code(language,problem):
    """
    Generates Python code for the given problem description.
    Args:
        problem (str): The problem description.
    Returns:
        str: The generated Python code.
    """
    # Initialize Groq LLM
    llm = create_chat_groq()
    # Fetch the code generation prompt
    code_prompt = code_generator_prompt()
    # Create a chain with the prompt and LLM
    chain = code_prompt | llm
    # Invoke the chain with the problem input
    response = chain.invoke({
        "programming_language":language,
        "problem": problem})
    return response.content