# Import necessary modules
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.youtube_tools import YouTubeTools
from phi.tools.dalle import Dalle
import openai
from dotenv import load_dotenv

load_dotenv()


# Web Searching Agent
web_search_agent = Agent(
    name="Web Searching Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tool_calls=True,
    markdown=True,
)

# Financial Data Agent
financed_search_agent = Agent(
    name="Financial Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=["Use tables for presenting data."],
    show_tool_calls=True,
    markdown=True,
)

# Image Generation Agent
image_agent = Agent(
    name="Image Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[Dalle()],
    description="An AI agent that generates images using DALL-E.",
    instructions="""When the user asks you to create an image:
    1. Use the `create_image` tool to generate the image.
    2. Retrieve the generated image URLs.
    3. Validate the retrieved URLs and display them.""",
    markdown=True,
    show_tool_calls=True,
)

# YouTube Timestamp Agent
yt_timestamp_agent = Agent(
    name="YouTube Timestamp Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YouTubeTools()],
    instructions=[
        "You are a YouTube agent. First check the length of the video. Then provide detailed timestamps for the video.",
        "Don't hallucinate timestamps.",
        "Return timestamps in the format: `[start_time, end_time, summary]`.",
    ],
    show_tool_calls=True,
)

# Team of agents for collaboration
agent_team = Agent(
    team=[web_search_agent, financed_search_agent, image_agent, yt_timestamp_agent],
    instructions=[
        "Always include sources.",
        "Use tables for presenting data.",
        "Create images when required.",
        "Provide accurate YouTube timestamps."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Utility function to generate and get image URLs
def generate_and_get_image_urls(agent, prompt):
  
    response = agent.print_response(prompt) 
    images_res = agent.get_images()  # Fetch generated images

    image_urls = []
    if images_res and isinstance(images_res, list):
        for img in images_res:
            img_url = getattr(img, "url", None)
            if img_url:
                image_urls.append(img_url)
    return image_urls

def call_agent(agent_name, prompt):

    if agent_name == "web_search":
        return web_search_agent.print_response(prompt)
    elif agent_name == "finance":
        return financed_search_agent.print_response(prompt)
    elif agent_name == "image":
        return generate_and_get_image_urls(image_agent, prompt)
    elif agent_name == "youtube":
        return yt_timestamp_agent.print_response(prompt)
    else:
        raise ValueError("Invalid agent name. Please choose from: web_search, finance, image, youtube.")

if __name__ == "__main__":

    # Web Search Agent
    # print("Web Search Response:")
    # print(call_agent("web_search", "Tell me about the history of Python programming."))

    # Financial Agent
    print("\nFinancial Data Response:")
    print(call_agent("finance", "Is SIP good or Lampsum?"))

    # Image Agent
    # print("\nImage Generation Response:")
    # image_urls = call_agent("image", "Create an image of a CARING GIRL")
    # print("Generated Image URLs:", image_urls)

    # # YouTube Timestamp Agent
    # print("\nYouTube Timestamp Response:")
    # print(call_agent("youtube", "Get timestamps for a video on Python basics."))
