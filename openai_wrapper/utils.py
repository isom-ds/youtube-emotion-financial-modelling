import asyncio
import json
from openai import AsyncOpenAI, OpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field
from tqdm import tqdm

from keys import openai_key
from modules.ai.identification import create_system_prompt, create_emotion_topic_model
from openai_wrapper.utils import fixErrorsOpenAI


def create_batch_payload(
    data: list,
    filename: str,
    search_term: str = "",
    model: str = "gpt-4.1-mini-2025-04-14",
    schema: str = create_emotion_topic_model(),
):
    """
    Create a batch payload for the OpenAI API.

    Args:
        data (list): The list of texts to be analysed.
        filename (str): The name of the file to be processed.
        system_prompt (str): The system prompt to be used for the API.
        model (str): The model to be used for the API.
        response_format (str): The response format for the API.

    Returns:
        list: The batch payload.
    """

    # Create the batch payload
    payload = []

    if model.startswith("text-embedding-"):
        for text in data:
            if isinstance(text["text"], str):
                payload.append(
                    {
                        "custom_id": text["id"],
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "input": text["text"],
                            "model": model,
                            "encoding_format": "float",
                            "dimensions": 256,
                        },
                    }
                )

    elif model.startswith("gpt-"):
        for text in data:
            if isinstance(text["text"], str):
                payload.append(
                    {
                        "custom_id": text["id"],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": create_system_prompt(search_term),
                                },
                                {"role": "user", "content": text["text"]},
                            ],
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "topic_emotion_analysis",
                                    "schema": schema,
                                    "strict": True,
                                },
                            },
                        },
                    }
                )
    else:
        raise ValueError(f"Unsupported model: {model}")

    with open(filename, "w") as file:
        for item in payload:
            json_line = json.dumps(item)
            file.write(json_line + "\n")


class FixedJsonResponse(BaseModel):
    fixed_json: str = Field(description="The corrected JSON object")


async def fixErrorsOpenAI(
    client: AsyncOpenAI,
    text: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    retries: int = 3,
) -> str:

    # Send to Open AI for processing
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant. Your task is to fix a json object with errors.
                    Only return the corrected JSON object.
                    Format as single line with no indents.
                    """,
                },
                {"role": "user", "content": f"{text}"},
            ],
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fixed_json",
                    "schema": to_strict_json_schema(FixedJsonResponse),
                    "strict": True,
                },
            },
        )

        output = json.loads(
            json.loads(chat_completion.choices[0].message.content)["fixed_json"]
        )

        return output
    except:
        if retries > 0:
            print("Retrying")
            output = await fixErrorsOpenAI(text=text, retries=retries - 1)
            return output
        else:
            return "Incorrect JSON format"


class FixedTopic(BaseModel):
    fixed_topic: str = Field(
        description="The corrected topic selected from damages, hurricane advice, hurricane relief services, personal opinion, weather information"
    )


def fixTopicErrorsOpenAI(
    text: str,
    topic_list: list,
    client: OpenAI = OpenAI(api_key=openai_key),
    model: str = "gpt-4o-mini-2024-07-18",
    retries: int = 3,
) -> str:

    if text.lower() not in topic_list:
        # Send to Open AI for processing
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are a helpful assistant. Your task is to categorise the words into one of the following categories: damages, hurricane advice, hurricane relief services, personal opinion, or weather information.
                        Only return one category.
                        """,
                    },
                    {"role": "user", "content": f"{text}"},
                ],
                model=model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fixed_topic",
                        "schema": to_strict_json_schema(FixedTopic),
                        "strict": True,
                    },
                },
            )

            output = json.loads(chat_completion.choices[0].message.content)[
                "fixed_topic"
            ]

            return output
        except:
            if retries > 0:
                print("Retrying")
                output = fixErrorsOpenAI(client=client, text=text, retries=retries - 1)
                return output
            else:
                return "Incorrect format"
    else:
        return text.lower()


async def jsonl2dict(data: list, client: AsyncOpenAI, batch_size: int = 2000) -> list:

    async def extractDataLine(json_object: list):
        if json_object["response"]["status_code"] != 200:
            content = "Failed Response"
            error = True

        else:
            content = json_object["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            try:
                content = json.loads(content)
                error = False
            except:
                # Try to fix JSON errors using OpenAI
                content = await fixErrorsOpenAI(client, content)
                error = True

        return {
            "batch": json_object["id"],
            "id": json_object["custom_id"],
            "content": content,
            "error": error,
        }

    # Process batches
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    output = []
    for batch in tqdm(
        batches, desc="Processing Batches", unit="batch", total=len(batches)
    ):
        batch_results = await asyncio.gather(*[extractDataLine(line) for line in batch])
        output.extend(batch_results)

    return output


def save_cleaned(data: list, path: str) -> list:
    # Fix additional errors
    print(f"{len([i for i in data if i['error']])} errors found")

    for i in data:
        if i["error"]:
            i["content"] = "Incorrect JSON format"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    return data
