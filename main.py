from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent,tool
import os
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                            temprature=0.7,)
search = TavilySearchResults(search_depth = 'basic')

@tool
def currentDate(format: str = "%d-%m-%Y"):
    """Returns the current date in the specified format."""
    today = datetime.datetime.now()
    formatted_date = today.strftime(format)
    return formatted_date

tools = [search, currentDate]


agent = initialize_agent(tools = tools, llm = llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("what was the last match Pakistan and India played and how many days ago it was from today?")

# result = llm.invoke("Tell me about todays weather in faisalabad")
# print(result)