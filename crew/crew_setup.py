"""
Crew Setup Module
Orchestrates the agents and tasks into a CrewAI process.
"""

from typing import Dict, Any
import os
from dotenv import load_dotenv

from crewai import Crew, Process
from langchain_groq import ChatGroq

from agents.intake_agent import create_intake_agent
from agents.retriever_agent import create_retriever_agent
from agents.rule_extractor_agent import create_rule_extractor_agent
from agents.explanation_agent import create_explanation_agent
from agents.verifier_agent import create_verifier_agent

from crew.tasks import (
    create_intake_task,
    create_retriever_task,
    create_rule_extraction_task,
    create_planner_task,
    create_verifier_task,
)

from tools.parser_tool import StudentProfileParserTool
from tools.vector_search_tool import VectorSearchTool
from tools.pdf_tool import PDFSearchTool
from rag.retriever import CatalogRetriever

load_dotenv()


class CoursePlannerCrew:
    """Orchestrates the Course Planning Assistant Crew."""
    
    def __init__(self, groq_api_key: str = None, groq_model: str = "llama-3.3-70b-versatile"):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided")
            
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=groq_model,
            temperature=0.1,
            max_tokens=4096,
        )
        
        # Initialize Core RAG components to share among tools
        self.retriever = CatalogRetriever()
        
        # Initialize Tools
        self.parser_tool = StudentProfileParserTool()
        self.vector_search_tool = VectorSearchTool(retriever=self.retriever)
        self.pdf_tool = PDFSearchTool()
        
        # Initialize Agents
        self.intake_agent = create_intake_agent(
            llm=self.llm, tools=[self.parser_tool]
        )
        self.retriever_agent = create_retriever_agent(
            llm=self.llm, tools=[self.vector_search_tool, self.pdf_tool]
        )
        self.rule_extractor_agent = create_rule_extractor_agent(
            llm=self.llm, tools=[]
        )
        self.planner_agent = create_explanation_agent(
            llm=self.llm, tools=[]
        )
        self.verifier_agent = create_verifier_agent(
            llm=self.llm, tools=[]
        )
        
    def run(self, user_query: str) -> str:
        """Run the crew to process a student query."""
        
        # Define tasks for this query
        intake_task = create_intake_task(self.intake_agent, user_query)
        retriever_task = create_retriever_task(self.retriever_agent, user_query)
        rule_task = create_rule_extraction_task(self.rule_extractor_agent)
        planner_task = create_planner_task(self.planner_agent, user_query)
        verifier_task = create_verifier_task(self.verifier_agent)
        
        # Setup task dependencies (Sequential)
        retriever_task.context = [intake_task]
        rule_task.context = [retriever_task]
        planner_task.context = [intake_task, rule_task, retriever_task]
        verifier_task.context = [planner_task]
        
        # Form the crew
        crew = Crew(
            agents=[
                self.intake_agent,
                self.retriever_agent,
                self.rule_extractor_agent,
                self.planner_agent,
                self.verifier_agent
            ],
            tasks=[
                intake_task,
                retriever_task,
                rule_task,
                planner_task,
                verifier_task
            ],
            process=Process.sequential,
            verbose=True,
        )
        
        # Execute workflow
        result = crew.kickoff()
        return result
