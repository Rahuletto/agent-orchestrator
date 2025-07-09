from agno.agent import Agent
from agno.models.google import Gemini

# Tools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.arxiv import ArxivTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.pubmed import PubmedTools
from agno.tools.website import WebsiteTools
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.crawl4ai import Crawl4aiTools
from agno.tools.pandas import PandasTools
from agno.tools.youtube import YouTubeTools

tools = [
    PandasTools(),
    Crawl4aiTools(max_length=None),
    ReasoningTools(add_instructions=True),
    YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        company_info=True,
        company_news=True,
    ),
    PubmedTools(),
    CalculatorTools(
        add=True,
        subtract=True,
        multiply=True,
        divide=True,
        exponentiate=True,
        factorial=True,
        is_prime=True,
        square_root=True,
    ),
    WikipediaTools(),
    ArxivTools(),
    WebsiteTools(),
    DuckDuckGoTools(),
    YouTubeTools(),
]


def CreateMaster(model, system, outputModel):
    agent = Agent(
        model=Gemini(id=model),
        tools=tools,
        system_message=system,
        show_tool_calls=True,
        instructions=[
            "You are very intelligent AI agent that supposed to mitigate the tasks that are necessary to the sub agents.",
            "Also if an agent, you can give 'relies_on' in json which relies on response of the agents so it will wait.",
            "Do not wrap the json with markdown",
        ],
        use_json_mode=True,
        response_model=outputModel,
    )

    return agent


def CreateChild(model, system):
    agent = Agent(
        model=Gemini(id=model),
        tools=tools,
        system_message=system,
        show_tool_calls=True,
        instructions=[
            "Use tables to display data.",
            "Be professional in your responses.",
            "You are supposed to give the best result no matter how long it takes",
        ],
        markdown=True,
    )

    return agent


def FinalVerdict(model, system):
    agent = Agent(
        model=Gemini(id=model),
        tools=tools,
        system_message=system,
        show_tool_calls=True,
        instructions=[
            "You can use tables to show data if necessary",
            "Be professional in your responses.",
            "You are supposed to give the best result no matter how long it takes",
        ],
        markdown=True,
    )

    return agent
