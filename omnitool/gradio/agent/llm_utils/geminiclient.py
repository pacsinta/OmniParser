from google import genai
from google.genai import types
import os
from .utils import is_image_path

def run_gemini_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=256, temperature=0.6):
    """
    Run a chat completion through Gemini's API, ignoring any images in the messages.
    """

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    
    # Configure the Gemini API
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    
    # Create properly formatted messages for Gemini
    json_messages = []
    
    # Add system message
    json_messages.append({"role": "user", "content": [system]})
    
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                # For dict items, concatenate all text content, ignoring images
                text_contents = []
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if not is_image_path(cnt):  # Skip image paths
                            text_contents.append(cnt)
                    else:
                        text_contents.append(str(cnt))
                
                if text_contents:  # Only add if there's text content
                    role = item.get("role", "user")
                    json_messages.append({"role": role, "content": [" ".join(text_contents)]})
            else:  # str
                json_messages.append({"role": "user", "content": [item]})
    
    elif isinstance(messages, str):
        json_messages.append({"role": "user", "content": [messages]})

    # convert json array to string array
    string_messages = []
    for message in json_messages:
        string_message = f"{message['role']}: {' '.join(message['content'])}"
        string_messages.append(string_message)
        print(f"Gemini message: {string_message}")

    try:
        # Create a conversation and send the messages
        response = client.models.generate_content(
            model=model_name,
            contents=string_messages,
            config=config
        )
        
        # Extract the response content
        content = response.text
        final_answer = content.split('</think>\n')[-1] if '</think>' in content else content
        final_answer = final_answer.replace("<output>", "").replace("</output>", "")
        
        # Gemini doesn't provide token usage in the same way as OpenAI
        # We'll estimate it based on input and output length
        total_input_chars = sum(len(str(msg.get("parts", [""])[0])) for msg in json_messages)
        token_usage = int((total_input_chars + len(content)) / 4)  # rough estimate
        
        return final_answer, token_usage
    except Exception as e:
        print(f"Error in Gemini API: {e}")
        return str(e), 0