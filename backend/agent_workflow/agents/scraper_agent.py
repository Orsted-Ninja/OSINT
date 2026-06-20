import os
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from tools.sherlock_tool import username_osint
from tools.profile_scraper import scrape_profile
from tools.holehe_tool import email_osint
from tools.dns_tool import dns_lookup
from tools.whois_tool import whois_lookup
from core.llm import get_llm

tools = [
    scrape_profile,
    username_osint,
    email_osint,
    dns_lookup,
    whois_lookup
]

def get_agent_executor(llm_model: str = None):
    llm = get_llm(model=llm_model)
    return create_react_agent(
        model=llm,
        tools=tools,
    )

def agent_invoke(inputs: dict, llm_model: str = None):
    from langchain_core.messages import SystemMessage
    system_prompt = """
    ROLE
    You are a web scraping agent.

    PRIMARY RESPONSIBILITY
    Use the available scraping tools to retrieve information from URLs provided by the user.

    RULES
    - Always use a scraping tool when a URL is provided.
    - Never invent webpage content.
    - Never answer from prior knowledge when a URL is supplied.
    - Return only information extracted from the source.
    - Preserve source attribution.
    - If extraction fails, report the tool error.
    - Do not summarize unless requested.
    - Do not analyze, classify, or infer.

    OUTPUT
    Return valid JSON whenever possible.
    """
    messages = [SystemMessage(content=system_prompt)] + inputs.get("messages", [])
    agent_executor = get_agent_executor(llm_model)
    return agent_executor.invoke({"messages": messages})

# Expose agent with a compatible invoke interface
class AgentWrapper:
    def invoke(self, inputs, llm_model: str = None):
        return agent_invoke(inputs, llm_model)

agent = AgentWrapper()
