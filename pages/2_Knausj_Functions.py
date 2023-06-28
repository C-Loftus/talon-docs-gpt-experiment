import streamlit as st
import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory


# import env from one level up
import sys
sys.path.append("..")
import env


os.environ['OPENAI_API_KEY'] = env.API_KEY.openai

# App framework
st.title('🦜 Knausj GPT Creator')
prompt = st.text_input('I want a Knausj Talon function that I can say in order to: ', placeholder="open a new tab") 

# Prompt templates
function_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Find Talon functions from the Knausj Talon repo that can be called in order to {topic}'
)


# Memory 
function_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0) 
function_chain = LLMChain(llm=llm, prompt=function_template, verbose=True, output_key='function', memory=function_memory)


# Show stuff to the screen if there's a prompt
if prompt: 
    result = function_chain.run(prompt)
    script = function_chain.run(function=result, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(function_memory.buffer)


    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)