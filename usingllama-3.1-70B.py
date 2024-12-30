# Import necessary libraries
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.youtube_tools import YouTubeTools
from phi.tools.dalle import Dalle


# Finance Agent
finance_agent = Agent(
    model=Groq(id="llama-3.1-70B-versatile"),
    instructions=["You are a financial assistant specializing in stock market data and analysis."]
)

# Web Search Agent
web_search_agent = Agent(
    model=Groq(id="llama-3.1-70B-versatile"),
    instructions=["Always include sources for your responses."],

)




# Unified Function to Call Agents
def call_agent(agent_name, prompt):

    if agent_name == "finance":
        return finance_agent.print_response(prompt)
    elif agent_name == "web_search":
        return web_search_agent.print_response(prompt)

  
    else:
        raise ValueError("Invalid agent name. Please choose from: finance, web_search, image, youtube.")



# Main Execution Block
if __name__ == "__main__":
    # print("Finance Agent Response:")
    # print(call_agent("finance", "What is the book value of stocks?"))

    print("\nWeb Search Agent Response:")
    print(call_agent("web_search", "What is love?"))

   