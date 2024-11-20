import language_tool_python
import json
import re

tool = language_tool_python.LanguageTool('en-US')

def check_grammar_with_rationale(text):
    # Check for grammar errors
    matches = tool.check(text)
    
    # Store the corrections
    corrections = []
    
    # Loop over all the grammar issues found
    for match in matches:
        # Extract the word index where the match occurs
        start_index = match.offset
        word_start = text.rfind(' ', 0, start_index) + 1
        word_end = text.find(' ', start_index)
        if word_end == -1:
            word_end = len(text)
        word = text[word_start:word_end]
        
        # Calculate the word index
        words = text[:word_start].split()
        word_index = len(words)  # Position of the current word in the text
        
        # Store the correction details
        corrections.append({
            "word_index": word_index,
            "original_word": word,
            "message": match.message,
            "replacements": match.replacements,
            "error_type": match.ruleId
        })
    
    # Return the results in JSON format
    return {
        "original_text": text,
        "corrections": corrections
    }