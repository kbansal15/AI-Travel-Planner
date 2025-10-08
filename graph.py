# graph.py

from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# Define the state for our graph
class TravelPlannerState(TypedDict):
    user_request: str
    research_info: str
    itinerary: str
    review: str
    messages: Annotated[List[BaseMessage], operator.add]

class TravelPlannerGraph:
    def __init__(self, agents):
        self.researcher = agents["researcher"]
        self.itinerary_agent = agents["itinerary_agent"]
        self.reviewer = agents["reviewer"]
        self.graph = self._build_graph()

    def _build_graph(self):
        # Initialize the state graph
        workflow = StateGraph(TravelPlannerState)

        # Define the nodes
        workflow.add_node("run_researcher", self.run_researcher)
        workflow.add_node("run_itinerary_agent", self.run_itinerary_agent)
        workflow.add_node("run_review", self.run_review)

        # Set the entry point
        workflow.set_entry_point("run_researcher")

        # Define the edges
        workflow.add_edge("run_researcher", "run_itinerary_agent")
        workflow.add_edge("run_itinerary_agent", "run_review")
        workflow.add_edge("run_review", END) # End the process after review

        # Compile the graph
        return workflow.compile()

    # Node functions
    def run_researcher(self, state):
        print("---RUNNING RESEARCHER---")
        user_request = state["user_request"]
        research_info = self.researcher.invoke({"input": user_request})
        return {"research_info": research_info['output']}

    def run_itinerary_agent(self, state):
        print("---RUNNING ITINERARY AGENT---")
        user_request = state["user_request"]
        research_info = state["research_info"]
        prompt = f"User Request: {user_request}\n\nResearch Information:\n{research_info}"
        itinerary = self.itinerary_agent.invoke({"input": prompt})
        return {"itinerary": itinerary['output']}

    def run_review(self, state):
        print("---RUNNING REVIEWER---")
        user_request = state["user_request"]
        itinerary = state["itinerary"]
        prompt = f"Original User Request: {user_request}\n\nGenerated Itinerary to Review:\n{itinerary}"
        review = self.reviewer.invoke({"input": prompt})
        # For simplicity, we'll just store the review. In a more complex app, you might loop back.
        return {"review": review['output']}

    def execute(self, user_request):
        initial_state = {"user_request": user_request}
        # The final result will be the state of the graph after execution
        final_state = self.graph.invoke(initial_state)
        return final_state
