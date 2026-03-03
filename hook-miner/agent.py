#!/usr/bin/env python3
"""
AI-driven web scraping agent powered by Claude.

Give it a URL, search query, or plain-language description and Claude will:
  1. Search and/or fetch the relevant pages
  2. Identify and extract structured data
  3. Save results to JSON and (where applicable) CSV

Usage:
    python agent.py "scrape the top 10 Python packages from pypi.org"
    python agent.py "https://news.ycombinator.com - get the front page stories"
    python agent.py  # prompts interactively

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY environment variable set
"""

import csv
import json
import sys

import anthropic

SYSTEM_PROMPT = """\
You are an expert web scraping and data extraction agent.

When given a task:
1. Use web_search to find relevant URLs if none are provided.
2. Use web_fetch to retrieve and read the pages.
3. Identify ALL structured data present (tables, lists, product listings, etc.).
4. Decide on a clean, consistent schema that captures every useful field.
5. When you have collected enough data, call save_results with:
   - filename: a short descriptive base name (no extension), e.g. "hn_stories"
   - records: an array of extracted record objects
   - summary: a brief description of what was collected and how many records

Follow links to additional pages if they contain more relevant data, but stay
focused on the original task. Prefer complete records over many shallow ones.\
"""


def save_to_disk(filename: str, records: list, summary: str) -> str:
    """Write records to <filename>.json and (if tabular) <filename>.csv."""
    json_path = f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    csv_path = None
    if records and isinstance(records[0], dict):
        fieldnames = list(dict.fromkeys(k for r in records for k in r.keys()))
        csv_path = f"{filename}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    msg = f"Saved {len(records)} records to {json_path}"
    if csv_path:
        msg += f" and {csv_path}"
    return msg


def run(task: str) -> None:
    client = anthropic.Anthropic()

    tools = [
        # Server-side tools — Claude calls these; the API handles execution
        {"type": "web_search_20260209", "name": "web_search"},
        {"type": "web_fetch_20260209", "name": "web_fetch"},
        # Client-side tool — we execute this when Claude calls it
        {
            "name": "save_results",
            "description": (
                "Save the extracted structured data to JSON and CSV files. "
                "Call this once when you have finished collecting all the data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Short descriptive base filename without extension, "
                            "e.g. 'hn_stories' or 'python_packages'"
                        ),
                    },
                    "records": {
                        "type": "array",
                        "description": "Array of extracted record objects",
                        "items": {"type": "object"},
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief human-readable summary of what was extracted",
                    },
                },
                "required": ["filename", "records", "summary"],
            },
        },
    ]

    messages: list = [{"role": "user", "content": task}]
    user_content = task
    max_continuations = 10
    continuation_count = 0

    print(f"Task: {task}\n")

    while True:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)

            response = stream.get_final_message()

        if response.stop_reason == "end_turn":
            print("\n\nDone.")
            break

        # Server-side tools hit their loop limit — re-send to continue
        if response.stop_reason == "pause_turn":
            if continuation_count >= max_continuations:
                print("\n\nReached maximum continuation limit.")
                break
            continuation_count += 1
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response.content},
            ]
            continue

        # Claude wants to call a user-defined tool
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "save_results":
                    result = save_to_disk(
                        block.input["filename"],
                        block.input["records"],
                        block.input["summary"],
                    )
                    print(f"\n{result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            continue

        break


def main() -> None:
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("What should I scrape? (URL or description): ").strip()

    if not task:
        print("No task provided.")
        sys.exit(1)

    try:
        run(task)
    except anthropic.AuthenticationError:
        print("\nError: Invalid API key. Set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("\nError: Could not connect to the Anthropic API.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
