import asyncio
import json
import traceback
from dotenv import load_dotenv
from typing import AsyncGenerator, Callable, Optional

import pydantic
from typedef.main import MasterAgentConfig
from utils.agent import CreateMaster, FinalVerdict, CreateChild
from agno.agent import RunResponse
from agno.utils.pprint import pprint_run_response
from collections import defaultdict
from pydantic import TypeAdapter


async def run_child_agent(
    agent_config, index, parent_results=None, stream_callback=None
):
    """
    Run a child agent with retry logic and essential streaming.
    """
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            final_prompt = agent_config.prompt
            parent_context = ""

            if parent_results and agent_config.relies_on:
                parent_contexts = []
                for parent_idx in agent_config.relies_on:
                    if parent_idx not in parent_results:
                        raise ValueError(
                            f"Agent {index} depends on Agent {parent_idx}, but its results are not available"
                        )

                    parent_config = parent_results[parent_idx]["config"]
                    parent_output = parent_results[parent_idx]["result"]

                    parent_contexts.append(
                        f"### Input from Agent {parent_idx} ({parent_config.type}) ###\n"
                        f"Focus: {parent_config.usecase}\n\n"
                        f"{parent_output}\n"
                        f"### End of input from Agent {parent_idx} ###"
                    )

                parent_context = "\n\n".join(parent_contexts)
                final_prompt = f"{parent_context}\n\n{agent_config.prompt}"

            if stream_callback and attempt == 0:
                await stream_callback(
                    {
                        "event": "agent_created",
                        "data": {
                            "agent_id": index,
                            "type": agent_config.type,
                            "usecase": agent_config.usecase,
                            "query": agent_config.prompt,
                            "depends_on": agent_config.relies_on,
                        },
                    }
                )

            if not stream_callback:
                print(f"[{index}] Creating and running agent: {agent_config.type}")
                if parent_results and agent_config.relies_on:
                    print(
                        f"[{index}] Added context from {len(agent_config.relies_on)} parent agent(s)"
                    )

            child_agent = CreateChild(
                model=agent_config.model,
                system=agent_config.system,
            )

            child_result = await child_agent.arun(final_prompt)

            if stream_callback:
                content = child_result.content
                chunk_size = 100

                for i, chunk_start in enumerate(range(0, len(content), chunk_size)):
                    chunk_end = min(chunk_start + chunk_size, len(content))
                    chunk = content[chunk_start:chunk_end]
                    is_final_chunk = chunk_end >= len(content)

                    await stream_callback(
                        {
                            "event": "agent_response_chunk",
                            "data": {
                                "agent_id": index,
                                "type": agent_config.type,
                                "content": chunk,
                                "is_final": is_final_chunk,
                            },
                        }
                    )

                    if not is_final_chunk:
                        await asyncio.sleep(0.03)

                await stream_callback(
                    {
                        "event": "agent_complete",
                        "data": {
                            "agent_id": index,
                            "type": agent_config.type,
                        },
                    }
                )

            if not stream_callback:
                print(f"\n--- Result from Agent {index} ({agent_config.type}) ---")
                pprint_run_response(child_result)

            return {
                "config": agent_config,
                "result": child_result.content,
                "index": index,
            }

        except Exception as e:
            error_msg = f"Agent {index} ({agent_config.type}) failed on attempt {attempt + 1}: {str(e)}"

            if stream_callback:
                await stream_callback(
                    {
                        "event": "error",
                        "data": {
                            "agent_id": index,
                            "type": agent_config.type,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error": str(e),
                        },
                    }
                )

            if not stream_callback:
                print(error_msg)

            if attempt < max_retries:
                if not stream_callback:
                    print(
                        f"[{index}] Retrying agent {agent_config.type} (attempt {attempt + 2}/{max_retries + 1})"
                    )

                await asyncio.sleep(1)
                continue
            else:
                fallback_response = f"Agent {agent_config.type} encountered technical difficulties after {max_retries + 1} attempts. Error: {str(e)}. Continuing with other agents."

                if stream_callback:
                    await stream_callback(
                        {
                            "event": "agent_failed",
                            "data": {
                                "agent_id": index,
                                "type": agent_config.type,
                                "error": str(e),
                                "fallback_response": fallback_response,
                            },
                        }
                    )

                if not stream_callback:
                    print(
                        f"[{index}] Agent {agent_config.type} failed after all retries. Using fallback response."
                    )

                return {
                    "config": agent_config,
                    "result": fallback_response,
                    "index": index,
                }


async def run_agents(query: str, stream_callback: Optional[Callable] = None):
    """
    Run the agent orchestration with essential streaming and error resilience.
    """
    load_dotenv()

    try:
        readFile = open("rao/prompts/system.txt", "r")
        system = readFile.read()
        readFile.close()
    except FileNotFoundError:
        readFile = open("./prompts/system.txt", "r")
        system = readFile.read()
        readFile.close()

    if system == "":
        raise ValueError("System file is empty. Please provide a valid system file.")

    master_agent = CreateMaster("gemini-2.0-flash", system, MasterAgentConfig)

    try:
        if not stream_callback:
            print("Running master agent...")
        master_result: RunResponse = master_agent.run(query)
        if not stream_callback:
            pprint_run_response(master_result)

        master_config = None
        if (
            isinstance(master_result.content, str)
            and "```json" in master_result.content
        ):
            if not stream_callback:
                print("Detected JSON in markdown format. Processing...")
            content = master_result.content

            json_start = content.find("```json") + 7
            json_end = content.rfind("```")
            json_str = content[json_start:json_end].strip()

            try:
                adapter = TypeAdapter(MasterAgentConfig)
                master_config = adapter.validate_json(json_str)
                if not stream_callback:
                    print("Successfully parsed JSON markdown into MasterAgentConfig")
            except json.JSONDecodeError as je:
                if not stream_callback:
                    print(f"Error parsing JSON: {je}")
                raise
            except Exception as e:
                if not stream_callback:
                    print(f"Error converting to MasterAgentConfig: {e}")
                raise
        else:
            master_config: MasterAgentConfig = master_result.content

        if not stream_callback:
            print(f"\nMaster agent created {len(master_config.agents)} child agents")
            print("Agent details:")
            for i, agent in enumerate(master_config.agents):
                print(
                    f"  Position {i}: Agent index={agent.id}, type={agent.type}, relies_on={agent.relies_on}"
                )

        results = {}

        agent_map = {}
        for agent in master_config.agents:
            agent_map[agent.id] = agent

        if not stream_callback:
            print(
                f"Created agent map with {len(agent_map)} entries: {list(agent_map.keys())}"
            )

        def is_ready(agent_idx):
            agent = agent_map[agent_idx]
            if not agent.relies_on:
                return True
            return all(parent_idx in results for parent_idx in agent.relies_on)

        remaining_agents = set(agent.id for agent in master_config.agents)
        if not stream_callback:
            print(f"Initial remaining agents: {remaining_agents}")

        while remaining_agents:
            ready_agents = [idx for idx in remaining_agents if is_ready(idx)]

            if not ready_agents:
                error_msg = "Error: Circular dependency detected or missing agents"
                if not stream_callback:
                    print(error_msg)
                    print(f"Remaining agents: {remaining_agents}")
                    for idx in remaining_agents:
                        agent = agent_map.get(idx)
                        if agent:
                            print(
                                f"Agent {idx} ({agent.type}) depends on: {agent.relies_on}"
                            )
                            for dep in agent.relies_on:
                                print(
                                    f"  - Dependency {dep} completed: {dep in results}"
                                )

                if stream_callback:
                    await stream_callback(
                        {
                            "event": "error",
                            "data": {"message": error_msg},
                        }
                    )
                break

            if not stream_callback:
                print(
                    f"\nRunning {len(ready_agents)} agents in parallel: {ready_agents}"
                )
            tasks = []
            for idx in ready_agents:
                agent = agent_map[idx]
                task = asyncio.create_task(
                    run_child_agent(agent, idx, results, stream_callback)
                )
                tasks.append(task)

            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    agent_idx = ready_agents[i]
                    agent = agent_map[agent_idx]
                    fallback_result = {
                        "config": agent,
                        "result": f"Agent {agent.type} encountered an unrecoverable error: {str(result)}. Continuing with other agents.",
                        "index": agent_idx,
                    }
                    results[agent_idx] = fallback_result
                    remaining_agents.remove(agent_idx)

                    if stream_callback:
                        await stream_callback(
                            {
                                "event": "agent_failed",
                                "data": {
                                    "agent_id": agent_idx,
                                    "type": agent.type,
                                    "error": str(result),
                                    "fallback_response": fallback_result["result"],
                                },
                            }
                        )
                else:
                    agent_idx = result["index"]
                    results[agent_idx] = result
                    remaining_agents.remove(agent_idx)
                    if not stream_callback:
                        print(
                            f"✓ Completed Agent {agent_idx} ({result['config'].type})"
                        )

        if not stream_callback:
            print(
                f"All agents completed. Results available for: {list(results.keys())}"
            )
        child_results = list(results.values())

        verdict_system = """
        You are LearnLM, an unbiased research and synthesis AI. Your task is to analyze all provided agent responses and create a comprehensive, unbiased final output that integrates all perspectives. Do not favor any specific agent or perspective. Present a balanced view that considers all input equally. Focus on factual information and clearly distinguish between consensus views and areas of disagreement. Do not add any personal opinions or biases. Your goal is to provide the most objective and comprehensive synthesis possible.
        """

        verdict_prompt = f"""
        ORIGINAL QUERY: {query}
        """

        for result in child_results:
            config = result["config"]
            content = result["result"]

            verdict_prompt += f"""
                AGENT: {config.type}
                FOCUS: {config.usecase}
                
                OUTPUT:
                {content}
                """

        verdict_prompt += """
        TASK: Analyze all agent responses provided above and produce a final, fully synthesized, actionable output that directly answers the original query with research evidences. Integrate all relevant information and perspectives from the agents equally—do not favor any single response. 
        Your response should not reflect on summary or the inputs—instead, deliver a clear, structured, and technically accurate final result as if you were the final decision-maker. Combine the best ideas, resolve overlaps or conflicts, and generate a unified, high-value deliverable for the user. 
        This is not a commentary—this is the final product.
        """

        if not stream_callback:
            print("\nCreating final verdict agent...")
        verdict_agent = FinalVerdict(
            model="learnlm-2.0-flash-experimental",
            system=verdict_system,
        )

        if stream_callback:
            await stream_callback(
                {
                    "event": "final_agent_start",
                    "data": {"message": "Initializing final synthesis agent"},
                }
            )

        final_result = await verdict_agent.arun(verdict_prompt)

        if stream_callback:
            content = (
                final_result.content
                if hasattr(final_result, "content")
                else str(final_result)
            )

            chunk_size = 150
            for i, chunk_start in enumerate(range(0, len(content), chunk_size)):
                chunk_end = min(chunk_start + chunk_size, len(content))
                chunk = content[chunk_start:chunk_end]
                is_final_chunk = chunk_end >= len(content)

                await stream_callback(
                    {
                        "event": "final_response_chunk",
                        "data": {
                            "content": chunk,
                            "is_final": is_final_chunk,
                        },
                    }
                )

                if not is_final_chunk:
                    await asyncio.sleep(0.05)

            await stream_callback(
                {
                    "event": "complete",
                    "data": {"message": "Orchestration complete"},
                }
            )
        else:
            verdict_agent.print_response(verdict_prompt, stream=True, markdown=True)

        return (
            final_result.content
            if hasattr(final_result, "content")
            else str(final_result)
        )

    except Exception as e:
        error_msg = f"Error running agent pipeline: {e}"
        print(error_msg)
        traceback.print_exc()

        if stream_callback:
            await stream_callback(
                {
                    "event": "error",
                    "data": {"message": error_msg, "traceback": traceback.format_exc()},
                }
            )
        raise
