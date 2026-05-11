import re
from pathlib import Path
import networkx as nx
import igraph as ig
import leidenalg
import json


def parse_markdown_file(path):
    text = Path(path).read_text(encoding="utf-8")

    # concept name = filename
    filename = Path(path).stem

    title = filename

    # extract links like [[concepts/xyz]]
    links = re.findall(r"\[\[concepts/(.*?)\]\]", text)
    # links = [name.replace("_", "-") for name in links]

    description = extract_description(text)

    return {"name": title, "links": links, "description": description}


def extract_description(text):
    # try YAML brief
    match = re.search(r"brief:\s*(.*)", text)
    if match:
        return match.group(1).strip()

    # fallback: first paragraph
    paragraphs = text.split("\n\n")
    return paragraphs[0][:200]


def load_all_markdown(concepts_dir):
    files = list(Path(concepts_dir).glob("*.md"))

    concepts = []

    for f in files:
        concepts.append(parse_markdown_file(f))
        print(f)
        print(parse_markdown_file(f))

    return concepts


def build_graph(concepts):
    G = nx.Graph()

    for c in concepts:
        G.add_node(c["name"])

    for c in concepts:
        src = c["name"]

        for tgt in c["links"]:
            G.add_edge(src, tgt)

    return G


def nx_to_igraph(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}

    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]

    g = ig.Graph(edges=edges, directed=False)
    g.vs["name"] = [reverse_mapping[i] for i in range(len(reverse_mapping))]

    return g


def leiden_partition(graph):
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)

    communities = []
    for cluster in partition:
        communities.append([graph.vs[i]["name"] for i in cluster])

    return communities


def hierarchical_leiden(graph, min_size=5, level=0):
    communities = leiden_partition(graph)

    # 🔴 STOP CONDITION 1: no split happened
    if len(communities) == 1:
        return [
            {
                "level": level,
                "nodes": [graph.vs[i]["name"] for i in range(graph.vcount())],
                "children": [],
            }
        ]

    result = []

    for comm in communities:
        # 🔴 STOP CONDITION 2: too small
        if len(comm) <= min_size:
            result.append({"level": level, "nodes": comm, "children": []})
            continue

        # build subgraph
        indices = [v.index for v in graph.vs if v["name"] in comm]
        subgraph = graph.subgraph(indices)

        # 🔴 STOP CONDITION 3: subgraph same size as parent
        if subgraph.vcount() == graph.vcount():
            result.append({"level": level, "nodes": comm, "children": []})
            continue

        # recurse safely
        children = hierarchical_leiden(subgraph, min_size=min_size, level=level + 1)

        result.append({"level": level, "nodes": comm, "children": children})

    return result


def hierarchy_to_tree(hierarchy):
    def convert(node):
        return {
            "name": f"L{node['level']} ({len(node['nodes'])})",
            "children": [convert(child) for child in node.get("children", [])],
            "members": node["nodes"],  # actual concepts
        }

    return [convert(root) for root in hierarchy]


def save_tree(hierarchy, descriptions, path="data/tree.json"):
    def enrich(node):
        return {
            "name": f"Level {node['level']}",
            "children": [enrich(child) for child in node.get("children", [])],
            "concepts": [
                {"name": n, "description": descriptions.get(n, "")}
                for n in node["nodes"]
            ],
        }

    tree = [enrich(root) for root in hierarchy]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)


def build_hierarchical_communities(G_ig):

    # Step 4: hierarchical Leiden
    hierarchy = hierarchical_leiden(G_ig, min_size=6)

    return hierarchy


def print_hierarchy(tree, indent=0):
    for node in tree:
        print("  " * indent + f"- Level {node['level']} ({len(node['nodes'])} nodes)")

        # print a few sample nodes
        for n in node["nodes"][:5]:
            print("  " * (indent + 1) + n)

        if node["children"]:
            print_hierarchy(node["children"], indent + 1)


def flatten_hierarchy(tree):
    communities = {}
    cid = 0

    def dfs(node):
        nonlocal cid
        communities[cid] = node["nodes"]
        cid += 1

        for child in node["children"]:
            dfs(child)

    for root in tree:
        dfs(root)

    return communities


def save_graph(G, descriptions, path="data/graph.json"):
    data = {"nodes": [], "edges": []}

    for node in G.nodes():
        data["nodes"].append({"id": node, "description": descriptions.get(node, "")})

    for u, v in G.edges():
        data["edges"].append({"source": u, "target": v})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_hierarchy_enriched(hierarchy, descriptions, path="data/hierarchy.json"):
    def enrich(node):
        return {
            "level": node["level"],
            "nodes": [
                {"name": n, "description": descriptions.get(n, "")}
                for n in node["nodes"]
            ],
            "children": [enrich(child) for child in node.get("children", [])],
        }

    data = [enrich(root) for root in hierarchy]

    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_all(G, hierarchy, descriptions, path):
    Path(path).mkdir(exist_ok=True)

    save_graph(G, descriptions, path=f"{path}/graph.json")

    save_hierarchy_enriched(hierarchy, descriptions, path=f"{path}/hierarchy.json")

    save_tree(hierarchy, descriptions, path=f"{path}/tree.json")

    print("✅ Saved graph, communities, hierarchy")


def main():

    # Step 1: load markdown
    concepts = load_all_markdown("wiki/concepts/")

    # Step 2: build graph (from links)
    G_nx = build_graph(concepts)

    # Step 3: convert to igraph
    G_ig = nx_to_igraph(G_nx)

    communities = build_hierarchical_communities(G_ig)

    # print_hierarchy(communities)

    # build description map
    descriptions = {c["name"]: c["description"] for c in concepts}

    communities_hier = flatten_hierarchy(communities)

    save_all(G_nx, communities, descriptions, "wiki/graph/")


if __name__ == "__main__":
    main()
