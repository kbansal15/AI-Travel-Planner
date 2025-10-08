AI Multi-Agent Travel Planner
This project is an autonomous travel planning system that uses a team of specialized AI agents to create personalized travel itineraries from a single user prompt. By orchestrating multiple agents, it automates the entire process of research, planning, and review, delivering a comprehensive travel plan in under a minute.

How It Works
The application uses a multi-agent architecture built with LangGraph, where each agent has a specific role:

Researcher Agent: Gathers up-to-date information on flights, accommodations, and attractions based on the user's request.

Itinerary Agent: Takes the researched data and constructs a detailed, day-by-day travel and budget plan.

Reviewer Agent: Critiques the generated plan for feasibility and coherence, ensuring it meets the user's original requirements.

Technology Stack
Agentic Workflow: LangGraph & LangChain

Language Model: Llama3 via Groq API (for high-speed inference)

User Interface: Streamlit

Programming Language: Python
