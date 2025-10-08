# main.py

import os
import streamlit as st
from dotenv import load_dotenv

# Since all files are in the same directory, we can use direct imports.
from agents import create_travel_planner_agents
from graph import TravelPlannerGraph

# Load environment variables from the .env file in the same directory
load_dotenv()

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="Multi-Agent Travel Planner", page_icon="✈️")
    st.title("✈️ Multi-Agent Travel Planner")
    st.markdown("Provide your travel details below, and our team of AI agents will craft the perfect itinerary for you.")

    # Get Groq API key from environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Please make sure it's set in your .env file.")
        return

    # User input form
    with st.form("travel_form"):
        user_request = st.text_area(
            "What are your travel plans?",
            "I want to plan a 7-day trip to Bali in December. I'm interested in beaches, culture, and hiking. My budget is around $2000.",
            height=150
        )
        submitted = st.form_submit_button("Plan My Trip")

    if submitted:
        if not user_request:
            st.warning("Please enter your travel plans.")
            return

        with st.spinner("🤖 Our AI agents are collaborating to build your itinerary... Please wait."):
            try:
                # Create the agents and the graph
                agents = create_travel_planner_agents(api_key)
                travel_graph = TravelPlannerGraph(agents)

                # Execute the plan
                result = travel_graph.execute(user_request)

                # Display the final itinerary
                st.subheader("Here is Your Final Travel Itinerary:")
                st.markdown(result["itinerary"])

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
