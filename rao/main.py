import asyncio
import json
import traceback
from dotenv import load_dotenv

import pydantic
from typedef.main import MasterAgentConfig
from utils.agent import CreateMaster, FinalVerdict, CreateChild
from agno.agent import RunResponse
from agno.utils.pprint import pprint_run_response
from collections import defaultdict
from pydantic import TypeAdapter


async def run_child_agent(agent_config, index, parent_results=None):
    """
    Run a child agent with context from parent agents if needed.

    Args:
        agent_config: The agent configuration
        index: The index of this agent
        parent_results: Dictionary of results from parent agents this agent depends on
    """
    print(f"[{index}] Creating and running agent: {agent_config.type}")

    final_prompt = agent_config.prompt

    # Include parent context if provided
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
        print(
            f"[{index}] Added context from {len(agent_config.relies_on)} parent agent(s)"
        )

    child_agent = CreateChild(
        model=agent_config.model,
        system=agent_config.system,
    )

    child_result = await child_agent.arun(final_prompt)

    print(f"\n--- Result from Agent {index} ({agent_config.type}) ---")
    pprint_run_response(child_result)

    return {"config": agent_config, "result": child_result.content, "index": index}


def query_input():
    """Get user query input with error handling"""
    return input("Enter your query: ")


async def main():
    load_dotenv()

    query = query_input()

    readFile = open("./prompts/system.txt", "r")
    system = readFile.read()
    if system == "":
        raise ValueError("System file is empty. Please provide a valid system file.")

    master_agent = CreateMaster("gemini-2.5-flash", system, MasterAgentConfig)

    try:
        print("Running master agent...")
        master_result: RunResponse = master_agent.run(query)
        pprint_run_response(master_result)

        # Parse master agent response
        master_config = None
        if (
            isinstance(master_result.content, str)
            and "```json" in master_result.content
        ):
            print("Detected JSON in markdown format. Processing...")
            content = master_result.content

            # Extract JSON from markdown code block
            json_start = content.find("```json") + 7
            json_end = content.rfind("```")
            json_str = content[json_start:json_end].strip()

            try:
                adapter = TypeAdapter(MasterAgentConfig)
                master_config = adapter.validate_json(json_str)
                print("Successfully parsed JSON markdown into MasterAgentConfig")
            except json.JSONDecodeError as je:
                print(f"Error parsing JSON: {je}")
                raise
            except Exception as e:
                print(f"Error converting to MasterAgentConfig: {e}")
                raise
        else:
            master_config: MasterAgentConfig = master_result.content

        print(f"\nMaster agent created {len(master_config.agents)} child agents")

        # Debug: Print agent information
        print("Agent details:")
        for i, agent in enumerate(master_config.agents):
            print(
                f"  Position {i}: Agent index={agent.id}, type={agent.type}, relies_on={agent.relies_on}"
            )

        # Store results by agent index
        results = {}

        # Create a mapping from agent index to the agent object
        agent_map = {}
        for agent in master_config.agents:
            agent_map[agent.id] = agent

        print(
            f"Created agent map with {len(agent_map)} entries: {list(agent_map.keys())}"
        )

        # Function to check if an agent is ready to run (all dependencies are completed)
        def is_ready(agent_idx):
            agent = agent_map[agent_idx]
            if not agent.relies_on:
                return True
            return all(parent_idx in results for parent_idx in agent.relies_on)

        # Get all indices of agents that need to be processed
        remaining_agents = set(agent.id for agent in master_config.agents)
        print(f"Initial remaining agents: {remaining_agents}")

        # Continue until all agents are processed
        while remaining_agents:
            # Find agents ready to run (all dependencies satisfied)
            ready_agents = [idx for idx in remaining_agents if is_ready(idx)]

            if not ready_agents:
                print("Error: Circular dependency detected or missing agents")
                print(f"Remaining agents: {remaining_agents}")
                for idx in remaining_agents:
                    agent = agent_map.get(idx)
                    if agent:
                        print(
                            f"Agent {idx} ({agent.type}) depends on: {agent.relies_on}"
                        )
                        for dep in agent.relies_on:
                            print(f"  - Dependency {dep} completed: {dep in results}")
                break

            # Run all ready agents concurrently
            print(f"\nRunning {len(ready_agents)} agents in parallel: {ready_agents}")
            tasks = []
            for idx in ready_agents:
                agent = agent_map[idx]
                task = asyncio.create_task(run_child_agent(agent, idx, results))
                tasks.append(task)

            # Wait for all tasks to complete
            completed_results = await asyncio.gather(*tasks)

            # Store results
            for result in completed_results:
                agent_idx = result["index"]
                results[agent_idx] = result
                remaining_agents.remove(agent_idx)
                print(f"✓ Completed Agent {agent_idx} ({result['config'].type})")

        # Prepare results for final verdict
        print(f"All agents completed. Results available for: {list(results.keys())}")
        child_results = list(results.values())

        # Create final verdict
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

        print("\nCreating final verdict agent...")
        verdict_agent = FinalVerdict(
            model="learnlm-2.0-flash-experimental",
            system=verdict_system,
        )

        verdict_agent.print_response(verdict_prompt, stream=True, markdown=True)

    except Exception as e:
        print(f"Error running agent pipeline: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
