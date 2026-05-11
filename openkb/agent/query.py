"""Q&A agent for querying the OpenKB knowledge base."""
from __future__ import annotations

from pathlib import Path

from agents import Agent, Runner, function_tool

from agents import ToolOutputImage, ToolOutputText
from openkb.agent.tools import get_wiki_page_content, read_wiki_file, read_wiki_image
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


import re
import os
import json

import time 
from datetime import datetime, timedelta
import requests

BASE_URL = "https://api.agriwatch.in/apgov"
HEADERS = {"Content-Type": "application/json"}
TOKEN_USERID = "apgov_agriwatch"
MAX_RETRIES = 3
RETRY_DELAY = 2 # seconds


MAX_TURNS = 50
from openkb.schema import get_agents_md

class AgriwatchClient:
    def __init__(self):
        self.token_key = None
        self.get_token()

    def get_token(self):
        url = f"{BASE_URL}/token.php"
        payload = {"token_userid": TOKEN_USERID}

        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()

        data = response.json()
        if data.get("status_code") != 200:
            raise Exception(f"Failed to get token: {data}")

        self.token_key = data["token_key"]
        print("🔑 New token generated")

    def call_api(self, endpoint, extra_payload=None):
        url = f"{BASE_URL}/{endpoint}"

        for attempt in range(MAX_RETRIES):
            payload = {"token_userid": TOKEN_USERID, "token_key": self.token_key}

            if extra_payload:
                payload.update(extra_payload)

            try:
                response = requests.post(url, headers=HEADERS, json=payload)

                # If unauthorized or token expired → regenerate token
                if response.status_code in [401, 403]:
                    print("⚠️ Token expired/invalid. Regenerating...")
                    self.get_token()
                    continue

                response.raise_for_status()
                data = response.json()

                # Optional: detect token failure via response body
                if isinstance(data, dict) and data.get("status") == "error":
                    if "token" in str(data).lower():
                        print("⚠️ Token issue in response. Regenerating...")
                        self.get_token()
                        continue

                return data

            except Exception as e:
                print(f"Retry {attempt + 1}/{MAX_RETRIES} failed: {e}")
                time.sleep(RETRY_DELAY)

        raise Exception(f"Failed API call after retries: {endpoint}")


_QUERY_INSTRUCTIONS_TEMPLATE = """\
You are OpenKB, a knowledge-base Q&A agent. You answer questions by searching the wiki.

{schema_md}

## Search strategy
1. Read index.md to see all documents and concepts with brief summaries.
   Each document is marked (short) or (pageindex) to indicate its type.
2. Read relevant summary pages (summaries/) for document overviews.
   Summaries may omit details — if you need more, follow the summary's
   `full_text` frontmatter field to the source (see step 4).
3. Read concept pages (concepts/) for cross-document synthesis.
4. When you need detailed source document content, each summary page has a
   `full_text` frontmatter field with the path to the original document content:
   - Short documents (doc_type: short): read_file with that path.
   - PageIndex documents (doc_type: pageindex): use get_page_content(doc_name, pages)
     with tight page ranges. The summary shows document tree structure with page
     ranges to help you target. Never fetch the whole document.
5. Source content may reference images (e.g. ![image](sources/images/doc/file.png)).
   Use the get_image tool to view them when needed.
6. For agricultural price questions:
    - use get_daily_prices
    - infer commodity/market/state from the user query
    - summarize trends clearly
7. Synthesize a clear, concise, well-cited answer grounded in wiki content.

Answer based only on wiki content. Be concise.
Before each tool call, output one short sentence explaining the reason.

If you cannot find relevant information, say so clearly.
"""

def _setup_llm_key(kb_dir: Path):
    """Set LiteLLM API key from LLM_API_KEY env var if present.

    Load order (override=False, so first one wins):
    1. System environment variables (already set)
    2. KB-local .env  (kb_dir/.env)
    3. Global .env    (~/.config/openkb/.env)

    Also propagates to provider-specific env vars (OPENAI_API_KEY, etc.)
    so that the Agents SDK litellm provider can pick them up.
    """
    import os
    from dotenv import dotenv_values
    env_file = os.path.join(kb_dir, ".env")

    config = dotenv_values(env_file)
    
    completion_kwargs = {}

    if "RITS_API_BASE" in config:
        completion_kwargs["RITS_API_BASE"] = config["RITS_API_BASE"]

    if "RITS_API_KEY" in config:
        completion_kwargs["RITS_API_KEY"] = config["RITS_API_KEY"]

    return completion_kwargs, config["RITS_MODEL"]


def build_query_agent(wiki_root: str, model: str, language: str = "en", kb_dir: str = ".openkb/") -> Agent:
    """Build and return the Q&A agent."""
    schema_md = get_agents_md(Path(wiki_root))
    instructions = _QUERY_INSTRUCTIONS_TEMPLATE.format(schema_md=schema_md)
    instructions += f"\n\nIMPORTANT: Answer in {language} language."

    SEARCH_DIRS = ["summaries", "concepts", "explorations"]
    documents = []
    doc_paths = []

    for folder in SEARCH_DIRS:
        folder_path = os.path.join(wiki_root, folder)

        if not os.path.exists(folder_path):
            continue

        for file_path in Path(folder_path).rglob("*.md"):
            try:
                text = file_path.read_text(encoding="utf-8")

                documents.append(text)
                doc_paths.append(file_path.relative_to(wiki_root).as_posix())

            except Exception:
                continue

    embedding_model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    document_embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    document_embeddings = np.array(document_embeddings)

    agri_client = AgriwatchClient()

    @function_tool(name_override="search")
    def search(query: str, top_k: int = 5) -> str:
        """
        Semantic vector search over the wiki.

        Use this tool first before read_file.

        Args:
            query: Search query.
            top_k: Number of results.
        """

        print(f"\n[SEARCH QUERY] {query}")

        # Embed query
        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        # Cosine similarity
        scores = cosine_similarity(
            [query_embedding],
            document_embeddings
        )[0]

        # Rank results
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in ranked_indices:

            score = scores[idx]

            path = doc_paths[idx]

            preview = documents[idx][:3000].replace("\n", " ")

            results.append(
                f"""
    FILE: {path}
    SIMILARITY: {score:.4f}

    DOCUMENT:
    {preview}
    """
            )

        if not results:
            return "No relevant documents found."

        return "\n\n".join(results)

    @function_tool
    def read_file(path: str) -> str:
        """Read a Markdown file from the wiki.
        Args:
            path: File path relative to wiki root (e.g. 'summaries/paper.md').
        """
        return read_wiki_file(path, wiki_root)

    @function_tool
    def get_page_content(doc_name: str, pages: str) -> str:
        """Get text content of specific pages from a PageIndex (long) document.
        Only use for documents with doc_type: pageindex. For short documents,
        use read_file instead.
        Args:
            doc_name: Document name (e.g. 'attention-is-all-you-need').
            pages: Page specification (e.g. '3-5,7,10-12').
        """
        return get_wiki_page_content(doc_name, pages, wiki_root)

    @function_tool
    def get_image(image_path: str) -> ToolOutputImage | ToolOutputText:
        """View an image from the wiki.

        Use when a question asks about a specific figure, chart, or diagram
        you'd need to see to answer accurately.

        Args:
            image_path: Image path relative to wiki root (e.g. 'sources/images/doc/p1_img1.png').
        """
        result = read_wiki_image(image_path, wiki_root)
        if result["type"] == "image":
            return ToolOutputImage(image_url=result["image_url"])
        return ToolOutputText(text=result["text"])

    @function_tool(name_override="get_daily_prices")
    def get_daily_prices(
        commodity: str = "",
        market: str = "",
        state: str = "",
        date: str = "",
        top_k: int = 20,
    ) -> str:
        """
        Fetch agricultural commodity daily prices from Agriwatch.

        Use this tool when the user asks about:
        - commodity prices
        - mandi prices
        - arrivals
        - price trends
        - market-wise prices
        - Andhra Pradesh agricultural prices

        Args:
            commodity: Commodity name like 'Black Gram', 'Cotton', 'Maize'
            market: Market name like 'Vijaywada'
            state: State name like 'Andhra Pradesh'
            date: Price date in YYYY-MM-DD format
            top_k: Maximum rows to return
        """

        # Default to today if not provided
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n[PRICE QUERY]")
        print(f"commodity={commodity}")
        print(f"market={market}")
        print(f"state={state}")
        print(f"date={date}")

        try:

            data = agri_client.call_api(
                "dailyPrice.php",
                {"priceDate": date}
            )

            rows = data.get("DATA_LIST", [])

            # ---------------------------------------------------
            # Filtering
            # ---------------------------------------------------

            filtered = []

            for row in rows:

                if commodity:
                    if commodity.lower() not in row.get(
                        "COMMODITY_NAME", ""
                    ).lower():
                        continue

                if market:
                    if market.lower() not in row.get(
                        "MARKET_NAME", ""
                    ).lower():
                        continue

                if state:
                    if state.lower() not in row.get(
                        "STATE_NAME", ""
                    ).lower():
                        continue

                filtered.append(row)

            # ---------------------------------------------------
            # Limit results
            # ---------------------------------------------------

            filtered = filtered[:top_k]

            if not filtered:
                return (
                    f"No price data found for "
                    f"commodity='{commodity}', "
                    f"market='{market}', "
                    f"state='{state}', "
                    f"date='{date}'"
                )

            # ---------------------------------------------------
            # Format results
            # ---------------------------------------------------

            formatted = []

            for item in filtered:

                formatted.append(
                    f"""
    COMMODITY: {item.get("COMMODITY_NAME")}
    VARIETY: {item.get("VARIETY_NAME")}
    STATE: {item.get("STATE_NAME")}
    MARKET: {item.get("MARKET_NAME")}

    CURRENT PRICE: {item.get("CURRENT_PRICE")} {item.get("PRICE_UNIT")}
    PREVIOUS PRICE: {item.get("PREVIOUS_PRICE")}

    PRICE CHANGE: {item.get("MODAL_PRICE_CHANGE")}

    CURRENT ARRIVAL: {item.get("CURRENT_ARRIVAL")} {item.get("ARRIVAL_UNIT")}
    PREVIOUS ARRIVAL: {item.get("PREVIOUS_ARRIVAL")}

    ARRIVAL CHANGE: {item.get("ARRIVALS_CHANGE")}

    SOURCE: {item.get("SOURCE")}
    """
                )

            return "\n\n".join(formatted)

        except Exception as e:

            return f"Failed to fetch daily prices: {str(e)}"


    from agents.model_settings import ModelSettings
    from openai import AsyncOpenAI
    from agents import OpenAIChatCompletionsModel

    completion_kwargs, model_name = _setup_llm_key(kb_dir)

    client = AsyncOpenAI(
        api_key="dummy",   # vLLM usually ignores this
        base_url=completion_kwargs["RITS_API_BASE"],
        default_headers={
            "RITS_API_KEY": completion_kwargs["RITS_API_KEY"]
        }
    )

    model = OpenAIChatCompletionsModel(
        model=model_name.split("hosted_vllm/")[-1],
        openai_client=client,
    )

    return Agent(
        name="wiki-query",
        instructions=instructions,
        tools=[read_file, get_page_content, get_image, search, get_daily_prices],
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=False, max_tokens=128000),
    )


async def run_query(
    question: str,
    kb_dir: Path,
    model: str,
    stream: bool = False,
    *,
    raw: bool = False,
) -> str:
    """Run a Q&A query against the knowledge base.

    Args:
        question: The user's question.
        kb_dir: Root of the knowledge base.
        model: LLM model name.
        stream: If True, print response tokens to stdout as they arrive.
        raw: If True, write raw markdown source instead of rendering it
            (still keeps tool-call line styling).

    Returns:
        The agent's final answer as a string.
    """
    import sys
    from agents import RawResponsesStreamEvent, RunItemStreamEvent, ItemHelpers
    from openai.types.responses import ResponseTextDeltaEvent
    from openkb.config import load_config

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    language: str = config.get("language", "en")

    wiki_root = str(kb_dir / "wiki")

    agent = build_query_agent(wiki_root, model, language=language)

    if not stream:
        result = await Runner.run(agent, question, max_turns=MAX_TURNS)
        return result.final_output or ""

    import os
    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR", "")

    from openkb.agent.chat import (
        _build_style,
        _fmt,
        _format_tool_line,
        _make_markdown,
        _make_rich_console,
    )

    style = _build_style(use_color)

    from rich.live import Live

    if use_color and not raw:
        console = _make_rich_console()
    else:
        console = None  # type: ignore[assignment]

    def _start_live() -> Live | None:
        if console is None:
            return None
        lv = Live(console=console, vertical_overflow="visible")
        lv.start()
        return lv

    live: Live | None = None
    last_was_text = False
    need_blank_before_text = False
    result = Runner.run_streamed(agent, question, max_turns=MAX_TURNS)
    collected: list[str] = []
    segment: list[str] = []
    try:
        live = _start_live()
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseTextDeltaEvent):
                    text = event.data.delta
                    if text:
                        if need_blank_before_text:
                            if console is not None:
                                print()
                                segment = []
                                live = _start_live()
                            else:
                                sys.stdout.write("\n")
                            need_blank_before_text = False
                        collected.append(text)
                        segment.append(text)
                        last_was_text = True
                        if live:
                            if "\n" in text:
                                joined = "".join(segment)
                                visible = joined[: joined.rfind("\n") + 1]
                                if visible:
                                    live.update(_make_markdown(visible))
                        else:
                            sys.stdout.write(text)
                            sys.stdout.flush()
            elif isinstance(event, RunItemStreamEvent):
                item = event.item
                if item.type == "tool_call_item":
                    if last_was_text:
                        if live:
                            if segment:
                                live.update(_make_markdown("".join(segment)))
                            live.stop()
                            live = None
                        else:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        last_was_text = False
                    raw_item = item.raw_item
                    name = getattr(raw_item, "name", "?")
                    args = getattr(raw_item, "arguments", "") or ""

                    try:
                        import json
                        parsed_args = json.loads(args)
                        pretty_args = json.dumps(parsed_args, indent=2)
                    except Exception:
                        pretty_args = str(args)
                    if live:
                        live.stop()
                        live = None
                    _fmt(style, ("class:tool", _format_tool_line(name, args) + "\n"))
                    need_blank_before_text = True
                elif item.type == "tool_call_output_item":
                    output = getattr(item, "output", None)

                    if output:
                        preview = str(output)

                        if len(preview) > 1000:
                            preview = preview[:1000] + "..."

                        _fmt(
                            style,
                            (
                                "class:tool",
                                f"\n[TOOL OUTPUT]\n{preview}\n"
                            ),
                        )
                    pass
    finally:
        if live:
            if segment:
                live.update(_make_markdown("".join(segment)))
            live.stop()
        print()
    return "".join(collected) if collected else result.final_output or ""
