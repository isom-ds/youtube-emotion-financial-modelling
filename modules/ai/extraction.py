def extract_topic(item: dict) -> set:
    if isinstance(item["content"], dict):
        if item["content"].get("emotopic"):
            return set(
                [
                    i["topic"]
                    .lower()
                    .replace("-", " ")
                    .replace("'s", "")
                    .replace("'", "")
                    .replace(".", "")
                    .replace("/", " ")
                    .replace('"', "")
                    .strip()
                    for i in item["content"]["emotopic"]
                    if len(i) > 0
                ]
            )


def extract_emotions(id: str, json_data: list):
    for item in json_data[:]:  # Iterate over a copy to avoid skipping elements
        if item["id"] == str(id):
            try:
                if isinstance(item["content"], dict):
                    output = sorted(
                        list(
                            set(
                                i
                                for sublist in item["content"]["emotions"]
                                for i in sublist
                            )
                        )
                    )
                else:
                    output = None

                # Remove processed item from list
                json_data.remove(item)

                return output
            except Exception as e:
                print(f"Error processing item: {item}, Error: {e}")

    return None  # Return None if no match is found


def extract_topics_json(id: str, json_data: list):
    for item in json_data[:]:  # Iterate over a copy to avoid skipping elements
        if item["id"] == str(id):
            try:
                if isinstance(item["content"], dict):
                    output = sorted(item["content"]["topics"])
                else:
                    output = None

                # Remove processed item from list
                json_data.remove(item)

                return output
            except Exception as e:
                print(f"Error processing item: {item}, Error: {e}")

    return None  # Return None if no match is found
