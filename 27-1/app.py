
from dotenv import load_dotenv
import streamlit  as st
import chain
load_dotenv()

def poem_generator_app():
    '''function returns the poem 
    '''
    with st.form("poemgenerator"):

        topic=st.text_input("Enter the topic for the poem")
        submitted=st.form_submit_button("submit")
        if(submitted):
            response = chain.generate_poem(topic)
            st.info(response)
poem_generator_app()
            
    
    
    