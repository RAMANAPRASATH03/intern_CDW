from dotenv import load_dotenv
import streamlit as st
import chain
# Load environment variables
load_dotenv()
def code_generator_app():
    """
    Streamlit app for generating Python code based on a problem description.
    """
    with st.form("codegenerator"):
        # Input field for the problem description
        language = st.text_input("Enter the programmimg language")
        problem = st.text_input("Enter the problem you want to solve:")
        submitted = st.form_submit_button("Generate Code")
        if submitted:
            if problem:
                # Generate Python code using the chain
                response = chain.generate_code(language,problem)
                st.info("Generated Code:")
                st.code(response, language="")
                
            else:
                st.warning("Please provide a problem description.")
# Run the Streamlit app
code_generator_app()