import json

def print_formatted_json(json_data: dict):
    """
    print formatted json data.
    """

    print(json.dumps(json_data, indent=4, ensure_ascii=False))
    print()