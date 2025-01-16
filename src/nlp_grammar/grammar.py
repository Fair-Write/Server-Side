import language_tool_python
import json

import language_tool_python

def check_text_with_language_tool(text: str) -> dict:
    # Initialize the language tool for English
    tool = language_tool_python.LanguageTool('en-US')

    # Analyze the text
    matches = tool.check(text)

    # Create a list for corrections
    corrections = []

    # Process each match to extract the necessary details
    for match in matches:
        corrections.append({
            "character_offset": match.offset,
            "character_endset": match.offset + match.errorLength,
            "original_text": text[match.offset:match.offset + match.errorLength],
            "message": match.message,
            "category": match.category,
            "rule_id": match.ruleId,
            "replacements": match.replacements
        })

    # Generate revised text with corrections applied
    revised_text = language_tool_python.utils.correct(text, matches)

    # Construct the final result
    result = {
        "original_text": text,
        "revised_text": revised_text,
        "corrections": corrections
    }

    return result

# Example Usage
input_text = "Weeed,, teacher is, teachhing."
output = check_text_with_language_tool(input_text)
print(output)
print(json.dumps(output, indent=4))
