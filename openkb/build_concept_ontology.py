

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import json
from tqdm import tqdm
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import litellm
import os

from dotenv import dotenv_values

def _setup_llm_key(kb_dir: Path):
    """Set LiteLLM API key from LLM_API_KEY env var if present.

    Load order (override=False, so first one wins):
    1. System environment variables (already set)
    2. KB-local .env  (kb_dir/.env)
    3. Global .env    (~/.config/openkb/.env)

    Also propagates to provider-specific env vars (OPENAI_API_KEY, etc.)
    so that the Agents SDK litellm provider can pick them up.
    """

    config = dotenv_values(kb_dir)

    completion_kwargs = {}

    if "RITS_API_BASE" in config:
        completion_kwargs["api_base"] = config["RITS_API_BASE"]

    if "RITS_API_KEY" in config:
        completion_kwargs["extra_headers"] = {
            "RITS_API_KEY": config["RITS_API_KEY"],
            "reasoning_effort": "high",  # key for deeper reasoning
        }

    return completion_kwargs, config["RITS_MODEL"]

def llm_call(prompt, completion_kwargs, model):
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        **completion_kwargs
    )

    return response["choices"][0]["message"]["content"]

def load_embedding_model():
    model = SentenceTransformer("BAAI/bge-m3")
    return model

def parse_concepts(index_path):
    text = Path(index_path).read_text(encoding="utf-8")

    concepts = []
    in_section = False

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("## Concepts"):
            in_section = True
            continue

        if line.startswith("## ") and in_section:
            break

        if in_section and line.startswith("-"):
            match = re.search(r"\[\[(.*?)\]\]", line)
            if not match:
                continue

            link = match.group(1)
            name = link.split("/")[-1]

            parts = line.split("—")
            desc = parts[1].strip() if len(parts) > 1 else ""

            concepts.append({
                "name": name,
                "description": desc
            })

    return concepts


def build_hierarchy_llm(concepts, completion_kwargs, model, chunk_size=25):
    chunks = [
        concepts[i:i+chunk_size]
        for i in range(0, len(concepts), chunk_size)
    ]

    partial_trees = []

    for chunk in tqdm(chunks):
        prompt = f"""
You are building a concept hierarchy.

Group concepts into parent-child relationships.

Return STRICT JSON:

{{
  "nodes": {{
    "concept": ["child1", "child2"]
  }},
  "roots": ["top-level concepts"]
}}

Concepts:
{json.dumps(chunk, ensure_ascii=False, indent=2)}
"""

        text = llm_call(prompt, completion_kwargs, model)

        try:
            parsed = json.loads(text)
            partial_trees.append(parsed)
            print(partial_trees)
        except:
            print("Failed chunk")
    
    try:
        merged_trees = merge_trees_llm(partial_trees, completion_kwargs, model)
        return merged_trees
    except:
        return partial_trees

def merge_trees_llm(trees, completion_kwargs, model):
    prompt = f"""
Merge multiple concept hierarchies into one.

Return STRICT JSON:
{{
  "nodes": {{ "concept": ["child"] }},
  "roots": []
}}

Trees:
{json.dumps(trees, indent=2)}
"""

    response = llm_call(prompt, completion_kwargs, model)

    return json.loads(response)

def embed_concepts(concepts, model):
    texts = [
        f"{c['name']}: {c['description']}"
        for c in concepts
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # IMPORTANT for cosine similarity
        show_progress_bar=True
    )

    return embeddings


def build_similarity_graph(concepts, embeddings, top_k=5, threshold=0.5):
    sim_matrix = cosine_similarity(embeddings)

    graph = {}

    for i, concept in enumerate(concepts):
        name = concept["name"]

        scores = []
        for j, score in enumerate(sim_matrix[i]):
            if i == j:
                continue
            if score >= threshold:
                scores.append((j, score))

        # sort by similarity
        scores = sorted(scores, key=lambda x: -x[1])[:top_k]

        neighbors = [concepts[j]["name"] for j, _ in scores]

        graph[name] = neighbors

    return graph

def build_tree_from_graph(graph):
    # count incoming edges
    incoming = {node: 0 for node in graph}

    for src, targets in graph.items():
        for t in targets:
            if t in incoming:
                incoming[t] += 1

    # roots = low incoming
    roots = sorted(incoming, key=lambda x: incoming[x])[:10]

    return {
        "nodes": graph,
        "roots": roots
    }

def print_tree(tree):
    nodes = tree["nodes"]

    def dfs(node, depth=0, visited=set()):
        if node in visited:
            return
        visited.add(node)

        print("  " * depth + "- " + node)

        for child in nodes.get(node, []):
            dfs(child, depth + 1, visited)

    for root in tree.get("roots", []):
        dfs(root)

def main():
    concepts = parse_concepts("wiki/index.md")

    print(f"Loaded {len(concepts)} concepts")

    # Choose mode
    USE_LLM = False

    if USE_LLM:
        completion_kwargs, model = _setup_llm_key(".openkb/.env")
        tree = build_hierarchy_llm(concepts, completion_kwargs, model)
        with open("wiki/hierachical_index.md", "w", errors="ignore", encoding="utf8") as writer:
            json.dump(tree, writer, indent=2, ensure_ascii=False)
    else:
        model = load_embedding_model()

        embeddings = embed_concepts(concepts, model)

        graph = build_similarity_graph(
            concepts,
            embeddings,
            top_k=5,
            threshold=0.55
        )

        tree = build_tree_from_graph(graph)


    print("\n=== CONCEPT TREE ===")
    print_tree(tree)

    with open("wiki/hierachical_index.md", "w", errors="ignore", encoding="utf8") as writer:
        json.dump(tree, writer, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
