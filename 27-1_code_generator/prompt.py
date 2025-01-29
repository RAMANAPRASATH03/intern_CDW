from langchain_core.prompts import ChatPromptTemplate
from langchain import hub


def code_generator_prompt():
    """
    Generates a prompt template for  code generation.
    Returns:
        ChatPromptTemplate: Configured ChatPromptTemplate instance.
    """
    system_msg = '''
    You are a  code generation assistant. Your task is to generate  code to solve the problem described by the user. Follow these guidelines:
    1. Only respond to queries explicitly requesting  code.
    2. The output must strictly be the  code itself, with no additional explanations or descriptions.
    3. If the query is unrelated to  code generation, respond with:
    "I am a  code generation assistant. Please ask me to generate  code."
    4. Ensure the generated code is functional and adheres to  best practices.
    '''
    user_msg = "Write a  code to solve the following problem:{programming_language} {problem}"
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_msg)
    ])
    return prompt_template


def code_generator_prompt_from_hub():
    """
    Fetches the code generation prompt template from LangSmith Hub.
    Returns:
        ChatPromptTemplate: ChatPromptTemplate instance pulled from LangSmith Hub.
     """
    prompt_template = hub.pull("poemgenerator/code_generator")
    return prompt_template