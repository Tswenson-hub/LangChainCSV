import streamlit as st

with st.sidebar:
    st.write("link to full dataset")
    st.markdown("https://www.kaggle.com/datasets/yasserh/walmart-dataset")

import streamlit as st
import pandas as pd
import os

st.title("LangChain + Walmart sales")
st.markdown("This is a simple agent powered by OPENAIs GPT 3.5. Using a CSV file containing walmart sales data, the user \
            can ask any question of the data and recieve an answer in plain text that is accurate based on the entire dataset. \
            A more powerful model could even product charts or pre-made PDFs on a daily basis for distribution. ")


st.subheader('app.py')
st.code(
"""
#!/usr/bin/env python3
import os

os.environ['OPENAI_API_KEY'] = "YOUR KEY HERE"
import pandas as pd
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.schema.language_model import BaseLanguageModel
import time
import streamlit as st
import random

csv_file_path = "Walmart.csv"
st.subheader("Small sample of Walmart sales data")
st.dataframe(pd.read_csv(csv_file_path).sample(50), width=1500)

st.info("What were walmarts total sales during this timeframe?")
st.info("What was the average temperature during this timeframe?")
st.info("What were the total sales on days where the temperature was below 60?")



def solve(query):
    llm=OpenAI(temperature=0)

    agent = create_csv_agent(
        llm,
        csv_file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    if query != None:
        answer = agent.invoke(query)
        st.write(answer)



query = st.chat_input("Enter prompt here:")
if query != None:
    solve(query)

"""
)



