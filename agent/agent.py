import os
import json
import logging
from datetime import datetime
from PyPDF2 import PdfReader

import genaihub_client

genaihub_client.set_environment_variables()
from gen_ai_hub.proxy import get_proxy_client
from gen_ai_hub.prompt_registry.client import PromptTemplateClient

from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add these imports for better formatting
from rich.console import Console

# At the top of your script, after imports:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(
    ROOT_DIR, "..", "data", "output", f"output_formatted_mcp_{timestamp}.json"
)
COOKBOOK_FILE = "AnalyticsToolkitCookBook.pdf"
TASK_FILE = "task.md"
RECURSION_LIMIT = 500

# Assign LLM with Generative AI Hub
from gen_ai_hub.proxy.langchain.init_models import init_llm
LLM = init_llm("anthropic--claude-4-sonnet")

# Initialize rich console for better printing
console = Console()


def serialize_message(msg):
    if hasattr(msg, "type") and hasattr(msg, "content"):
        return {"type": msg.type, "content": msg.content}
    return str(msg)


def save_output(messages):
    serializable_messages = [serialize_message(m) for m in messages]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        ROOT_DIR, "..", "data", "output", f"output_sac_beautifier_{ts}.json"
    )
    with open(output_path, "w") as f:
        json.dump(serializable_messages, f, indent=4)
    return serializable_messages


# Retrieve agent prompt from prompt registry on SAP AI Core
def retrieve_agent_prompt(tools):
    # proxy_client = get_proxy_client(proxy_version="gen-ai-hub")
    # prompt_registry_client = PromptTemplateClient(proxy_client=proxy_client)
    # response = prompt_registry_client.fill_prompt_template_by_id(
    #     template_id="c0b92dd9-cfd2-49d8-bfe0-7c65a9c2f9eb",
    #     input_params={"tool_names": tools},
    # )
    # return response.parsed_prompt[0].content
    cookbook_path = os.path.join(ROOT_DIR, "..", "data", "context", COOKBOOK_FILE)
    if COOKBOOK_FILE.lower().endswith(".pdf"):
        reader = PdfReader(cookbook_path)
        cookbook = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(cookbook_path, "r", encoding="utf-8", errors="replace") as f:
            cookbook = f.read()
    
    return f"""You are a helpful agent that is an expert in designing and improving 
    analytics dashboards using SAC. You have this cookbook to help you: \n{cookbook}.
    To open SAC (browser based tool) and improve the dashboard you have these tools available:\n{tools}"""

async def main():
    logger.info("Starting SAC Beautifier Agent main function.")
    logger.info("Importing LangChainAdapter and creating adapter")

    # Set up MCP client for Playwright MCP Server - Browser Automation
    server_params = StdioServerParameters(
        command="npx", args=["@playwright/mcp@latest"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Create LangChainAdapter and load tools
            logger.info("LangChainAdapter created. Creating tools...")
            tools = await load_mcp_tools(session)
            tool_names = "\n".join(
                [f"- {tool.name}: {tool.description}" for tool in tools]
            )
            logger.info(f"Tools created: {tools}")

            # Load task in markdown
            task_path = os.path.join(
                ROOT_DIR, "..", "data", "context", TASK_FILE
            )
            with open(task_path, "r") as f:
                task = f.read()
            logger.info("Tutorial loaded.")

            # Initialize the prebuilt agent from LangGraph
            logger.info("LLM initialized and create_react_agent.")
            agent = create_react_agent(LLM, tools)

            console.print(
                "[bold magenta]🤖 Starting SAC Beautifier Agent[/bold magenta]"
            )
            console.print(f"[bold green]Available Tools:[/bold green]\n{tool_names}")
            console.print(f"[bold green]Tutorial File:[/bold green] {TASK_FILE}")

            # Retrieve agent system prompt
            agent_system_prompt = retrieve_agent_prompt(tool_names)

            logger.info("Starting agent stream processing.")
            async for chunk in agent.astream(
                {
                    "messages": [
                        {"role": "system", "content": agent_system_prompt},
                        {
                            "role": "user",
                            "content": f"""Task in Markdown: {task}""",
                        },
                    ]
                },
                stream_mode="updates",
                config={"recursion_limit": RECURSION_LIMIT},
            ):
                for step, data in chunk.items():
                    # Log and print each step
                    logger.info(f"step: {step}")
                    console.print(f"step: {step}")
                    content = data["messages"][-1].content
                    console.print(f"content: {data['messages'][-1].content}")

                    # Prepare the output dict
                    output_dict = {"step": step, "content": content}
                    # Append to file as a JSON line
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(output_dict, ensure_ascii=False))
                        f.write("\n")


if __name__ == "__main__":
    import asyncio

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create handlers
    file_handler = logging.FileHandler("tutorial_tester_agent.log")
    stream_handler = logging.StreamHandler()
    # Create formatters and add it to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("Logger initialized. Starting main execution.")
    logger.info("Initializing MCP client and LLM.")
    try:
        logger.info("Running main async function.")
        asyncio.run(main())
        logger.info("Main execution finished successfully.")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
