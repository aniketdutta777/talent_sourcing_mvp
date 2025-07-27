import json
import os
import uuid
import chromadb
import io
import fitz # PyMuPDF
from openai import OpenAI
from anthropic import Anthropic
import random
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# (Global variables and all initialization/helper functions remain the same)
# ...

def search_lark_database(user_query: str, num_profiles_to_retrieve: int) -> dict:
    print("--- Firing search against Lark's Database ---")
    system_message = """You are an expert HR recruitment assistant... (rest of your detailed prompt)"""
    messages = [{"role": "user", "content": user_query}]
    try:
        response = global_client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.0, tools=[resume_search_tool_schema], messages=messages, system=system_message)
        
        if response.stop_reason == "tool_use":
            tool_use = next((block for block in response.content if block.type == "tool_use"), None)
            if not tool_use: return {"status": "error", "message": "Claude indicated tool use, but no tool was specified."}
            
            tool_output = resume_search_tool(**tool_use.input)

            if (isinstance(tool_output, list) and len(tool_output) > 0 and
                tool_output[0].get("message", "").startswith("No candidates found")):
                print("Tool returned no candidates. Bypassing final LLM analysis.")
                return {
                    "status": "success",
                    "analysis_data": {
                        "overall_summary": "The initial search did not find any relevant candidates in the database for this query.",
                        "candidates": [],
                        "overall_recommendation": "Try broadening your search terms."
                    }, "usage": {"input_tokens": 0, "output_tokens": 0}
                }

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(tool_output)}]})
            
            final_response = global_client_anthropic.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, temperature=0.5, messages=messages, system=system_message)
            json_string = final_response.content[0].text.strip().lstrip("```json").rstrip("```")
            parsed_json = json.loads(json_string)
            usage_data = {"input_tokens": final_response.usage.input_tokens, "output_tokens": final_response.usage.output_tokens}
            return {"status": "success", "analysis_data": parsed_json, "usage": usage_data}

        # --- NEW FIX: Handle cases where the AI responds directly ---
        elif response.stop_reason == "end_turn":
            print("Claude provided a direct response without tool use.")
            return {
                "status": "success",
                "analysis_data": {
                    "overall_summary": f"The AI provided a direct response: {response.content[0].text}",
                    "candidates": [],
                    "overall_recommendation": "No candidates were searched as the AI answered directly."
                },
                "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            }
        # --- END FIX ---

    except Exception as e:
        if "Expecting value" in str(e):
             return {"status": "error", "message": "LLM returned an empty or invalid response after tool use."}
        return {"status": "error", "message": f"LLM analysis failed: {e}"}
        
    # --- NEW FIX: Catch-all to prevent returning None ---
    return {"status": "error", "message": f"Unexpected response from Claude with stop reason: {response.stop_reason}"}

# (The rest of your core_logic.py file, including search_google_drive and the main
#  perform_claude_search_with_tool router, remains the same. It is omitted here for brevity.)