# agents.py

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

def create_travel_planner_agents(api_key):
    """
    Creates and returns the researcher, itinerary, and reviewer agents.
    This function ensures all necessary agents for the graph are available.
    """
    # Initialize the LLM with the Groq API key
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # Define tools - all agents will share the same search tool for simplicity
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    
    # --- Base Prompt Template for all agents ---
    # This provides the core structure the agent needs to follow.
    base_prompt_template = """
You are an AI assistant. Your role is to: {instructions}

You have access to the following tools:
{tools}

Use the following format for your response:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: [Your final, comprehensive response here]

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

    # --- Researcher Agent ---
    researcher_instructions = "You are a world-class travel researcher. Your job is to find the most relevant and up-to-date information for a user's travel query. Focus on finding practical details about flights, accommodations, attractions, and local customs. Your final answer should be a summary of all the key information found."
    researcher_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template=base_prompt_template,
        partial_variables={"instructions": researcher_instructions, "tools": str(tools), "tool_names": ", ".join([t.name for t in tools])}
    )
    researcher_agent = create_react_agent(llm, tools, researcher_prompt)
    researcher_executor = AgentExecutor(
        agent=researcher_agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True # Handle cases where the agent outputs a non-standard response
    )

    # --- Itinerary Agent ---
    itinerary_instructions = "You are a specialized travel itinerary planner. You will receive researched information and a user's request. Your goal is to craft a detailed, day-by-day travel plan. The itinerary must be logical, enjoyable, and account for travel time. Your final answer must be only the complete, well-formatted itinerary."
    itinerary_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template=base_prompt_template,
        partial_variables={"instructions": itinerary_instructions, "tools": str(tools), "tool_names": ", ".join([t.name for t in tools])}
    )
    itinerary_agent = create_react_agent(llm, tools, itinerary_prompt)
    itinerary_executor = AgentExecutor(
        agent=itinerary_agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )

    # --- Reviewer Agent ---
    reviewer_instructions = "You are an expert travel plan reviewer. Your task is to critique a given itinerary for feasibility, completeness, and alignment with the user's original request. Provide constructive feedback. If the plan is good, approve it with a concluding sentence. If it needs changes, provide specific, actionable suggestions. Your final answer should be the complete review."
    reviewer_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template=base_prompt_template,
        partial_variables={"instructions": reviewer_instructions, "tools": str(tools), "tool_names": ", ".join([t.name for t in tools])}
    )
    reviewer_agent = create_react_agent(llm, tools, reviewer_prompt)
    reviewer_executor = AgentExecutor(
        agent=reviewer_agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )

    return {
        "researcher": researcher_executor,
        "itinerary_agent": itinerary_executor,
        "reviewer": reviewer_executor,
    }

