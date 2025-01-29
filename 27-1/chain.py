from model import create_chat_groq
import prompt

def generate_poem(topic):
    llm=create_chat_groq()
    '''
    function generates poem
    
    arguments:
    topic(str)-topic of the poem

    Returns :
    response.content(str)        
        '''
    poem_prompt = prompt.poem_generator_prompt()
    llm = create_chat_groq()
    chain = poem_prompt | llm
    response = chain.invoke({
        "topic": topic
        })
    return response.content

