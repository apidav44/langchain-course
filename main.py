from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class WeatherReport(BaseModel):
    location: str = Field(description="City and country, for example Nantes, France")
    date: str = Field(description="Date of the weather report, format YYYY-MM-DD if available")
    time: str = Field(description="Time of observation or search, if available")
    temperature: str = Field(description="Current temperature, with unit")
    condition: str = Field(description="Current weather condition")
    source: str = Field(description="Source URL or source name")
    confidence: str = Field(description="High, medium, or low")
    note: str = Field(description="Important caveat, especially if time is not available")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)

agent = create_agent(
    model=llm,
    tools=[tavily_search],
    response_format=WeatherReport,
    system_prompt=(
        "You are a helpful assistant. "
        "Use Tavily search for current weather information. "
        "Only use information supported by search results. "
        "Return the final answer as structured data matching the required schema. "
        "If the exact observation time is not available, set time to 'not available' "
        "and explain it in the note field."
    ),
)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Utilise Tavily pour chercher la météo actuelle à Nantes, France. "
                        "Indique la date, l'heure si disponible, la température, "
                        "les conditions météo et la source."
                    )
                )
            ]
        },
        config={"recursion_limit": 8},
    )

    structured_report = result["structured_response"]

    print("=== Objet Pydantic ===")
    print(structured_report)

    print("\n=== JSON ===")
    print(structured_report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()