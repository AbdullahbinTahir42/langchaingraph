from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
import os


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.7,) 


server_params = StdioServerParameters(
    command='npx',
    env={
        'FIRECRAWL_API_KEY': os.getenv('FIRECRAWL_API_KEY', ''),
    },
    args=["firecrawl-mcp"]
)

async def main():
    async with stdio_client(server_params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            tools = load_mcp_tools(session)
            agent = create_react_agent(llm, tools, verbose=True)

            messages = [
                "role":"system","content":"You are a helpful assistant that can scrape websites, crawl pages, and extract data using firecrawl tools.think step by step and use appropriate tools to get the information required to answer the user's question.",
                ]

            print("Available tools:",*[tool.name for tool in tools], sep="\n - ")
            while True:
                user_input = input("User: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break
                response = await agent.arun(user_input)
                print(f"Agent: {response}")
