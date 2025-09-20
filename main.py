import operator
import time
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START
from langchain_core.tools import tool
import asyncio
import logging

# Web voyager
from web_voyager import graph
from playwright.async_api import async_playwright
from IPython import display
import base64


logger = logging.getLogger(__name__)

llm = ChatOllama(model="gpt-oss:20b", temperature=0)

search = DuckDuckGoSearchRun()

@tool
async def search_tool(query: str) -> str:
    """Call to search  query"""
    browser = await async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.

    browser = await browser.chromium.launch(headless=False, args=['--disable-blink-features=AutomationControlled',
            '--disable-web-security',
            '--disable-dev-shm-usage',
            '--no-sandbox',
            '--disable-gpu',
            '--disable-dev-tools',
            '--disable-software-rasterizer',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding'])

    time.sleep(3)
    page = await browser.new_page()
    _ = await page.goto("https://www.google.com")

    async def call_agent(question: str, page, max_steps: int = 150):
        event_stream = graph.astream(
            {
                "page": page,
                "input": question,
                "scratchpad": [],
            },
            {
                "recursion_limit": max_steps,
            },
        )
        final_answer = None
        steps = []
        async for event in event_stream:
            # We'll display an event stream here
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            display.clear_output(wait=False)
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            display.display(display.Image(base64.b64decode(event["agent"]["img"])))
            if "ANSWER" in action:
                final_answer = action_input[0]
                break
        return final_answer

    res = await call_agent(query, page)

    return res


# print(agent_executor.invoke({"messages": [("user", "Hello")]}))

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

# State
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given keyword, come up with 2 simple step by step search goal to gain information about the keyword. \
This plan should involve individual research objective, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
.Make sure that each step has all the information needed 
- do not skip steps.
- do not answer the questions
- do not add complex task such as crosscheck, verify, consult, etc.


{json_schema}\n Answer using this JSON-Schema.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOllama(
    model="gpt-oss:20b", temperature=0
)

# output = planner.invoke({"messages": [
#             ("user", "what is the hometown of the current Australia open winner?")
#         ],
#                          "json_schema": Plan.model_json_schema()})
# print(output.content)


async def execute_step(state: PlanExecute):
    # Base instruction for the CRA agent
    base_prompt = (
        "You are a helpful assistant. Use the provided context from past steps first "
        "to answer the query or execute the task. Only call the browsing/search tool "
        "if the context does not contain enough information to answer the task. "
        "Do not overly search the web; prefer using local context and the plan."
    )

    # Extract plan & task
    plan = state.get("plan", [])

    if not plan:
        raise ValueError("No plan found in state.")

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    print(f"\nCurrent Task: {task}")

    task_formatted = f"""For the following plan:
{plan_str}

You are tasked with executing step 1: {task}
"""

    # --- NEW: incorporate past_steps into prompt/context ---
    # state["past_steps"] expected format: [("query", "answer"), ...]
    raw_past = state.get("past_steps", []) or []
    # Keep only answers and ensure they're strings
    answers = []
    for item in raw_past:
        try:
            # tuple of (query, answer) or similar
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                ans = item[1]
            else:
                # fallback if stored differently
                ans = item
            # if ans is a dict or message object, try to extract .content
            if hasattr(ans, "content"):
                ans = ans.content
            answers.append(str(ans))
        except Exception:
            # skip problematic entries but keep going
            continue

    # Optionally limit how many past steps we pass (e.g., latest 10)
    MAX_CONTEXT = 10
    recent_answers = answers[-MAX_CONTEXT:]

    # Basic deduping (preserve order)
    seen = set()
    deduped_answers = []
    for a in recent_answers:
        if a not in seen:
            deduped_answers.append(a)
            seen.add(a)

    # Build a markdown-like context block to pass to agent
    if deduped_answers:
        context_block = "Context from previous steps (most recent first):\n\n"
        for i, ans in enumerate(reversed(deduped_answers), 1):  # show most recent first
            context_block += f"{i}. {ans}\n\n"
    else:
        context_block = "No previous step answers available.\n\n"

    # Combine base prompt and context into the "system" style message
    full_system_prompt = f"{base_prompt}\n\n{context_block}"

    # Create the CRA agent (you can move this agent creation outside this function
    # if you want to avoid recreating it every call)
    agent_executor = create_react_agent(llm, tools=[search_tool], prompt=full_system_prompt)

    # Now invoke the agent with the task as a user message
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    print(f"Agent Executor: {agent_response["messages"][-1].content}")

    # Extract final assistant content safely
    try:
        assistant_content = agent_response["messages"][-1].content
    except Exception:
        # Fallback if structure differs
        assistant_content = str(agent_response)

    # Return past_steps containing the executed task + agent answer
    # If you want to append to previous history instead of replacing, do that upstream.
    return {
        "past_steps": [(task, assistant_content)],
    }



async def replan_step(state: PlanExecute):
    """
    Replan by removing completed tasks from the plan and updating the state
    """
    current_plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    # Extract completed task names from past_steps
    completed_tasks = []
    for step_tuple in past_steps:
        if len(step_tuple) >= 1:
            completed_tasks.append(step_tuple[0])  # First element is the task

    # Remove completed tasks from the current plan
    updated_plan = []
    for task in current_plan:
        # Check if this task (or a similar one) has been completed
        task_completed = False
        for completed_task in completed_tasks:
            # Simple string matching - you might want to make this more sophisticated
            if task.lower().strip() == completed_task.lower().strip():
                task_completed = True
                break

        if not task_completed:
            updated_plan.append(task)

    # If no tasks remain, we might need to create a final response step
    if not updated_plan:

        return {"response":"All task executed successfully."}

    print(f"Remaining Plan: {updated_plan}")
    # Return the updated state
    return {
        "plan": updated_plan
    }

async def plan_step(state: PlanExecute):
    """Create initial plan"""
    try:
        # Use the actual input from state instead of hardcoded question
        planner = planner_prompt | ChatOllama(
            model="gpt-oss:20b", temperature=0
        )


        response = planner.invoke({
            "messages": [
                ("user", f"{state['input']}")
            ],
            "json_schema": Plan.model_json_schema()
        })

        # Extract plan steps from the response
        if hasattr(response, 'content'):
            # If response has structured content, extract it
            plan_data = response.content
            if isinstance(plan_data, dict) and 'steps' in plan_data:
                plan_steps = plan_data['steps']
            elif isinstance(plan_data, str):
                # Try to parse as JSON if it's a string
                import json
                try:
                    parsed_data = json.loads(plan_data)
                    plan_steps = parsed_data.get('steps', [])
                except json.JSONDecodeError:
                    # If parsing fails, create default steps
                    plan_steps = [
                        f"Understand the question: {state['input']}",
                        "Search for relevant information",
                        "Analyze and synthesize the findings",
                        "Provide a comprehensive answer"
                    ]
            else:
                plan_steps = []
        else:
            # If response doesn't have expected structure, use the response directly
            plan_steps = response if isinstance(response, list) else []

    except Exception as e:
        print(f"Error creating plan: {e}")
        # Ultimate fallback with more specific steps
        plan_steps = [
            "Search for relevant information",
            "Process and verify the information",
            "Formulate a complete answer"
        ]

    # Ensure we have valid plan steps
    if not plan_steps or not isinstance(plan_steps, list):
        plan_steps = [
            f"Research the question: {state['input']}",
            "Gather necessary information",
            "Provide comprehensive answer"
        ]

    print(f"Agent Planner: {plan_steps}")
    return {
        "plan": plan_steps,
        "current_step": 0
    }

async def should_end(state: PlanExecute):
    if "response" in state and state["response"]:

        if "past_steps" in state and state["past_steps"]:
            past_steps = state.get("past_steps", [])

            # Extract only the answers
            answers = [answer for _, answer in past_steps]

            # Join them as context
            context = "\n".join([f"- {a}" for a in answers])

            # Define prompt for report generation
            prompt_template = ChatPromptTemplate.from_template(
                """You are a helpful assistant. 
Generate a structured markdown report summarizing the following answer, only summarize the important topic and fact, give a title too:

{context}

Format it in markdown with headings, bullet points, and summary sections."""
            )

            chain = prompt_template | ChatOllama(model="gpt-oss:20b")  # adjust model name
            report = chain.invoke({"context": context})

            # Debug/Print final report
            print(report.content)

            report_md = report.content

            filename = "report.md"
            # optionally make filename dynamic, e.g. include timestamp
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_md)

        return END
    else:
        return "agent"


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")


# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

graph_runnable = workflow.compile()
#
# async def run_app():
#
#     config = {"recursion_limit": 50}
#     inputs = {"input": "what is the hometown of Ezra Cornell?"}
#     async for message_chunk, metadata in graph_runnable.astream(inputs, config=config, stream_mode="messages"):
#         if message_chunk.content:
#             print(message_chunk.content, end=" ", flush=True)
#
# # Run the async function
# asyncio.run(run_app())