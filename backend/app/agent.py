"""
Google Gemini Agent integration with Context Graph tools.
Provides function-calling tools for querying and updating the context graph.
"""

import json
from typing import Any

from google import genai
from google.genai import types

from .config import config
from .context_graph_client import context_graph_client
from .gds_client import gds_client
from .vector_client import vector_client


def slim_properties(props: dict) -> dict:
    """Remove large properties to reduce response size."""
    slim = {}
    for key, value in props.items():
        # Skip embedding vectors
        if key in ("fast_rp_embedding", "reasoning_embedding", "embedding"):
            continue
        # Truncate long strings
        if isinstance(value, str) and len(value) > 200:
            slim[key] = value[:200] + "..."
        # Limit list sizes
        elif isinstance(value, list) and len(value) > 10:
            slim[key] = value[:10]
        else:
            slim[key] = value
    return slim


def get_graph_data_for_entity(entity_id: str, depth: int = 2, limit: int = 30) -> dict:
    """Get graph visualization data centered on an entity."""
    try:
        graph_data = context_graph_client.get_graph_data(
            center_node_id=entity_id, depth=depth, limit=limit
        )
        # Build nodes list first
        nodes = [
            {
                "id": node.id,
                "labels": node.labels,
                "properties": slim_properties(node.properties),
            }
            for node in graph_data.nodes
        ]

        # Create set of node IDs for filtering relationships
        node_ids = {node["id"] for node in nodes}

        # Only include relationships where both nodes exist
        relationships = [
            {
                "id": rel.id,
                "type": rel.type,
                "startNodeId": rel.start_node_id,
                "endNodeId": rel.end_node_id,
                "properties": slim_properties(rel.properties),
            }
            for rel in graph_data.relationships
            if rel.start_node_id in node_ids and rel.end_node_id in node_ids
        ]

        return {
            "nodes": nodes,
            "relationships": relationships,
        }
    except Exception as e:
        print(f"Error getting graph data for entity {entity_id}: {e}")
        return {"nodes": [], "relationships": []}


# ============================================
# SYSTEM PROMPT
# ============================================

CONTEXT_GRAPH_SYSTEM_PROMPT = """You are an AI assistant for a financial institution with access to a Context Graph.

The Context Graph stores decision traces - the reasoning, context, and causal relationships behind every significant decision made in the organization. This enables you to:

1. **Find Precedents**: Search for similar past decisions to inform current recommendations
2. **Trace Causality**: Understand how past decisions influenced subsequent outcomes
3. **Record Decisions**: Create new decision traces with full reasoning context
4. **Detect Patterns**: Identify fraud patterns and entity duplicates using graph structure

## Key Concepts

**Event Clock vs State Clock**:
- Traditional systems store the "state clock" - what is true right now
- The Context Graph stores the "event clock" - what happened, when, and with what reasoning

**Decision Traces**:
- Every significant decision is recorded with full reasoning
- Risk factors, confidence scores, and applied policies are captured
- Causal chains show how decisions influenced each other

## Guidelines

When helping users:
1. **Always search for precedents** before making recommendations
2. **Explain your reasoning thoroughly** - this becomes part of the decision trace
3. **Cite specific past decisions** when they inform your recommendation
4. **Flag exceptions or escalations** that may be needed
5. **Consider both structural and semantic similarity** when finding related cases

You have access to tools that leverage both:
- **Semantic similarity** (text embeddings) - for matching by meaning
- **Structural similarity** (FastRP graph embeddings) - for matching by relationship patterns

This combination provides insights that are impossible with traditional databases."""


# ============================================
# TOOL DEFINITIONS (Gemini Function Declarations)
# ============================================


TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_customer",
        description="Search for customers by name, email, or account number. Returns customer profiles with risk scores and related account counts.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Search query string"),
                "limit": types.Schema(type="INTEGER", description="Max results to return", default=10),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_customer_decisions",
        description="Get all decisions made about a specific customer, including approvals, rejections, escalations, and exceptions.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "customer_id": types.Schema(type="STRING", description="The customer ID"),
                "decision_type": types.Schema(type="STRING", description="Filter by decision type"),
                "limit": types.Schema(type="INTEGER", description="Max results", default=20),
            },
            required=["customer_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="find_similar_decisions",
        description="Find structurally similar past decisions using FastRP graph embeddings. Returns decisions with similar influences, causes, and precedents as well as decisions about related accounts.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "decision_id": types.Schema(type="STRING", description="The internal decision ID (decision.id)"),
                "limit": types.Schema(type="INTEGER", description="Number of similar decisions to return", default=5),
            },
            required=["decision_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="find_precedents",
        description="Find precedent decisions that could inform the current decision. Uses both semantic similarity (meaning) and structural similarity (graph patterns).",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "scenario": types.Schema(type="STRING", description="Description of the current scenario"),
                "category": types.Schema(type="STRING", description="Decision category filter"),
                "limit": types.Schema(type="INTEGER", description="Max results", default=5),
            },
            required=["scenario"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_causal_chain",
        description="Trace the causal chain of a decision - what caused it and what it led to. Useful for understanding decision impact and history.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "decision_id": types.Schema(type="STRING", description="The decision ID"),
                "direction": types.Schema(type="STRING", description="'upstream', 'downstream', or 'both'", default="both"),
                "depth": types.Schema(type="INTEGER", description="Max depth to traverse", default=3),
            },
            required=["decision_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="record_decision",
        description="Record a new decision with full reasoning context. Creates a decision trace in the context graph that can be referenced by future decisions.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "decision_type": types.Schema(type="STRING", description="Type of decision"),
                "category": types.Schema(type="STRING", description="Decision category"),
                "reasoning": types.Schema(type="STRING", description="Full reasoning text"),
                "customer_id": types.Schema(type="STRING", description="Related customer ID"),
                "account_id": types.Schema(type="STRING", description="Related account ID"),
                "risk_factors": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="List of risk factors"),
                "precedent_ids": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="IDs of precedent decisions"),
                "confidence_score": types.Schema(type="NUMBER", description="Confidence score 0-1", default=0.8),
            },
            required=["decision_type", "category", "reasoning"],
        ),
    ),
    types.FunctionDeclaration(
        name="detect_fraud_patterns",
        description="Analyze accounts or transactions for potential fraud patterns using graph structure analysis. Checks an account's proximity to flagged transactions as well as the prevalence of flagged transactions in the community of related accounts.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "account_id": types.Schema(type="STRING", description="The internal account ID (account.id), not the customer-facing account number"),
                "neighbor_count": types.Schema(type="INTEGER", description="Number of example decisions to return from the community", default=5),
            },
            required=["account_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="find_decision_community",
        description="Find decisions in the same community using Leiden community detection. Returns decisions that are structurally related through causal chains and precedent relationships.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "decision_id": types.Schema(type="STRING", description="The internal decision ID (decision.id)"),
                "example_count": types.Schema(type="INTEGER", description="Number of example decisions to return from the community", default=5),
            },
            required=["decision_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="find_accounts_with_high_shared_transaction_volume",
        description="Find accounts that share high transaction volumes with a given account.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "account_id": types.Schema(type="STRING", description="The internal account ID (account.id), not the customer-facing account number"),
            },
            required=["account_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_policy",
        description="Get the current policy rules for a specific category. Returns policy details including thresholds and requirements. If policy_name is provided, returns policies matching any words in the name.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "category": types.Schema(type="STRING", description="Policy category"),
                "policy_name": types.Schema(type="STRING", description="Policy name to search for"),
            },
            required=[],
        ),
    ),
    types.FunctionDeclaration(
        name="execute_cypher",
        description="Execute a read-only Cypher query against the context graph for custom analysis. Only SELECT/MATCH queries are allowed.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "cypher": types.Schema(type="STRING", description="The Cypher query to execute"),
            },
            required=["cypher"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_schema",
        description="Get the graph database schema including node labels, relationship types, property keys, indexes, and constraints. Also returns counts for each node label and relationship type.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
            required=[],
        ),
    ),
]


# ============================================
# TOOL EXECUTION
# ============================================


def merge_graph_data(graphs: list[dict], max_nodes: int = 50, max_rels: int = 75) -> dict:
    """Merge multiple graph data objects, removing duplicates and limiting size."""
    all_nodes = {}
    all_relationships = {}

    for graph in graphs:
        if not graph:
            continue
        for node in graph.get("nodes", []):
            if len(all_nodes) < max_nodes:
                all_nodes[node["id"]] = node
        for rel in graph.get("relationships", []):
            # Only include relationships where both nodes are in the graph
            if rel.get("startNodeId") in all_nodes and rel.get("endNodeId") in all_nodes:
                if len(all_relationships) < max_rels:
                    all_relationships[rel["id"]] = rel

    return {
        "nodes": list(all_nodes.values()),
        "relationships": list(all_relationships.values()),
    }


async def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name and return the result."""
    try:
        if name == "search_customer":
            results = context_graph_client.search_customers(
                query=args["query"], limit=args.get("limit", 10)
            )
            graphs = []
            for customer in results[:3]:
                customer_id = customer.get("id")
                if customer_id:
                    customer_graph = get_graph_data_for_entity(customer_id, depth=1)
                    graphs.append(customer_graph)
            graph_data = merge_graph_data(graphs) if graphs else {"nodes": [], "relationships": []}
            return {"customers": results, "graph_data": graph_data}

        elif name == "get_customer_decisions":
            results = context_graph_client.get_customer_decisions(
                customer_id=args["customer_id"],
                decision_type=args.get("decision_type"),
                limit=args.get("limit", 20),
            )
            graph_data = get_graph_data_for_entity(args["customer_id"], depth=2)
            return {"decisions": results, "graph_data": graph_data}

        elif name == "find_similar_decisions":
            decision_id = args["decision_id"]
            limit = int(args.get("limit", 10))
            similar_decisions = gds_client.find_similar_decisions(decision_id, limit=limit)
            graph_data = get_graph_data_for_entity(decision_id, depth=2)
            return {"similar_decisions": similar_decisions, "graph_data": graph_data}

        elif name == "find_precedents":
            results = vector_client.find_precedents_hybrid(
                scenario=args["scenario"],
                category=args.get("category"),
                limit=args.get("limit", 5),
            )
            graph_data = None
            if results and len(results) > 0:
                first_id = results[0].get("id") if isinstance(results[0], dict) else None
                if first_id:
                    graph_data = get_graph_data_for_entity(first_id, depth=2)
            return {"precedents": results, "graph_data": graph_data}

        elif name == "get_causal_chain":
            results = context_graph_client.get_causal_chain(
                decision_id=args["decision_id"],
                direction=args.get("direction", "both"),
                depth=args.get("depth", 3),
            )
            graph_data = get_graph_data_for_entity(args["decision_id"], depth=3)
            return {"causal_chain": results, "graph_data": graph_data}

        elif name == "record_decision":
            reasoning_embedding = None
            try:
                reasoning_embedding = vector_client.generate_embedding(args["reasoning"])
            except Exception:
                pass
            decision_id = context_graph_client.record_decision(
                decision_type=args["decision_type"],
                category=args["category"],
                reasoning=args["reasoning"],
                customer_id=args.get("customer_id"),
                account_id=args.get("account_id"),
                risk_factors=args.get("risk_factors", []),
                precedent_ids=args.get("precedent_ids", []),
                confidence_score=args.get("confidence_score", 0.8),
                reasoning_embedding=reasoning_embedding,
            )
            return {
                "success": True,
                "decision_id": decision_id,
                "message": f"Decision recorded successfully with ID {decision_id}",
            }

        elif name == "detect_fraud_patterns":
            neighbor_count = int(args.get("neighbor_count", 5))
            results = gds_client.detect_fraud_patterns(
                account_id=args.get("account_id"),
                neighbor_count=neighbor_count,
            )
            return results

        elif name == "find_decision_community":
            decision_id = args["decision_id"]
            example_count = int(args.get("example_count", 5))
            results = gds_client.get_decision_community(
                decision_id=decision_id, example_count=example_count
            )
            graph_data = get_graph_data_for_entity(decision_id, depth=2)
            return {"community_decisions": results, "graph_data": graph_data}

        elif name == "find_accounts_with_high_shared_transaction_volume":
            results = gds_client.find_accounts_with_high_shared_transaction_volume(
                account_id=args.get("account_id")
            )
            return results

        elif name == "get_policy":
            policies = context_graph_client.get_policies(category=args.get("category"))
            if args.get("policy_name"):
                stop_words = {"the", "a", "an", "for", "and", "or", "of", "in", "to", "with"}
                search_words = [
                    word.lower()
                    for word in args["policy_name"].split()
                    if word.lower() not in stop_words and len(word) > 2
                ]
                scored_policies = []
                for policy in policies:
                    policy_name_lower = policy.get("name", "").lower()
                    matches = sum(1 for word in search_words if word in policy_name_lower)
                    if matches > 0:
                        scored_policies.append({"policy": policy, "relevance_score": matches})
                scored_policies.sort(key=lambda x: x["relevance_score"], reverse=True)
                if scored_policies:
                    return {
                        "matching_policies": [
                            {**sp["policy"], "relevance_score": sp["relevance_score"]}
                            for sp in scored_policies
                        ],
                        "search_terms": search_words,
                        "total_matches": len(scored_policies),
                    }
                else:
                    return {
                        "matching_policies": [],
                        "search_terms": search_words,
                        "total_matches": 0,
                        "all_policies_in_category": policies,
                        "note": f"No policies matched '{args['policy_name']}'. Showing all policies in category.",
                    }
            return policies

        elif name == "execute_cypher":
            return context_graph_client.execute_cypher(cypher=args["cypher"])

        elif name == "get_schema":
            return context_graph_client.get_schema()

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}


# ============================================
# AGENT CONTEXT
# ============================================

AVAILABLE_TOOLS = [
    "search_customer",
    "get_customer_decisions",
    "find_similar_decisions",
    "find_precedents",
    "get_causal_chain",
    "record_decision",
    "detect_fraud_patterns",
    "find_decision_community",
    "find_accounts_with_high_shared_transaction_volume",
    "get_policy",
    "execute_cypher",
    "get_schema",
]


def get_agent_context() -> dict[str, Any]:
    """Get agent context information for transparency/debugging."""
    return {
        "system_prompt": CONTEXT_GRAPH_SYSTEM_PROMPT,
        "model": config.gemini.chat_model,
        "available_tools": AVAILABLE_TOOLS,
    }


from .gemini_pool import gemini_pool


# ============================================
# AGENT SESSION MANAGEMENT
# ============================================


class ContextGraphAgent:
    """Wrapper for managing Gemini Agent sessions with function calling."""

    def __init__(self):
        self.model = config.gemini.chat_model
        self.tools = types.Tool(function_declarations=TOOL_DECLARATIONS)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def query(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a query to the agent and get the response with automatic key failover."""

        # Build contents from conversation history
        contents = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

        # Add current message
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

        tool_calls = []
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            response = gemini_pool.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=CONTEXT_GRAPH_SYSTEM_PROMPT,
                    tools=[self.tools],
                    temperature=0.7,
                ),
            )

            # Check if there are function calls
            function_calls = []
            text_parts = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                    if part.text:
                        text_parts.append(part.text)

            if not function_calls:
                # No more function calls — we have the final response
                response_text = "\n".join(text_parts)
                return {
                    "response": response_text,
                    "tool_calls": tool_calls,
                    "decisions_made": [],
                }

            # Add the model's response to the conversation
            contents.append(response.candidates[0].content)

            # Execute each function call and add results
            function_response_parts = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                tool_calls.append({"name": tool_name, "input": tool_args})

                result = await execute_tool(tool_name, tool_args)
                result_str = json.dumps(result, indent=2, default=str)

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result_str},
                    )
                )

            contents.append(types.Content(role="user", parts=function_response_parts))

        # If we exceeded max iterations, return what we have
        return {
            "response": "I encountered an issue processing your request. Please try again.",
            "tool_calls": tool_calls,
            "decisions_made": [],
        }

    async def query_stream(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ):
        """Send a query to the agent and stream the response with automatic key failover."""

        # Emit agent context first
        yield {"type": "agent_context", "context": get_agent_context()}

        # Build contents from conversation history
        contents = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

        tool_calls = []
        decisions_made = []
        max_iterations = 10

        for _ in range(max_iterations):
            response = gemini_pool.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=CONTEXT_GRAPH_SYSTEM_PROMPT,
                    tools=[self.tools],
                    temperature=0.7,
                ),
            )

            function_calls = []
            text_parts = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                    if part.text:
                        text_parts.append(part.text)

            # Stream text parts
            for text in text_parts:
                yield {"type": "text", "content": text}

            if not function_calls:
                break

            # Add model response to conversation
            contents.append(response.candidates[0].content)

            # Execute function calls
            function_response_parts = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                tool_call = {"name": tool_name, "input": tool_args}
                tool_calls.append(tool_call)

                yield {"type": "tool_use", **tool_call}

                result = await execute_tool(tool_name, tool_args)
                result_str = json.dumps(result, indent=2, default=str)

                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "output": result,
                }

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result_str},
                    )
                )

            contents.append(types.Content(role="user", parts=function_response_parts))

        # Final event with summary
        yield {
            "type": "done",
            "tool_calls": tool_calls,
            "decisions_made": decisions_made,
        }
