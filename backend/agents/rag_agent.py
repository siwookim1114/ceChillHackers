"""
RAG Agent
Handles: document ingestion (PDF -> S3 -> Bedrock KnowledgeBase) + context retrieval with citations
Called by: Professor Agent, TA Agent, Manager Agent (Core Engine)
STRICT: never returns raw original documents - summarized context only
"""
import json
import uuid
import base64
import boto3
from typing import TypedDict, Annotated, Optional

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain.chat_models import init_chat_model
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.memory import AgentCoreMemorySaver
from bedrock_agentcore.runtime.context import RequestContext

from config.config_loader import config