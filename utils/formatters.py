import json


# Split a list into chunks
def split_list_into_chunks(data, chunk_size=50000):
    """
    Splits a list into chunks of specified size.

    Args:
        data (list): The list to split.
        chunk_size (int): Number of elements per chunk.

    Returns:
        List of chunks (lists).
    """
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def find_duplicates_set(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)


def remove_duplicates_dicts(data):
    seen = set()
    unique = []
    for item in data:
        item_str = json.dumps(item, sort_keys=True)  # Convert dict to string
        if item_str not in seen:
            seen.add(item_str)
            unique.append(item)
    return unique


# Convert the response content (JSONL) to a list of dictionaries
def jsonl_to_list(response):
    # Decode the response content if it's in bytes
    content = response.text if hasattr(response, "text") else response.decode("utf-8")

    # Split by newlines and load each line as a dictionary
    return [json.loads(line) for line in content.strip().split("\n") if line]


# Load JSONL files
def load_jsonl(paths: str) -> list:

    output = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            jsonl_data = [json.loads(line) for line in file]
        output.extend(jsonl_data)

    return output


# Load JSON files
def load_json(path: str) -> list:

    with open(path, "r", encoding="utf-8") as file:
        output = json.load(file)

    return output
