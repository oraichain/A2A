import pytest
import json
from rewoo import split_large_result
from transformers import GPT2TokenizerFast
from tokenizers import Tokenizer
# Import your split_large_result here
# from your_module import split_large_result

# Setup tokenizer globally
tokenizer: Tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
MAX_TOKENS = 100  # Set small for easier testing

def is_wrapped_in_result_tags(chunk: str) -> bool:
    return chunk.strip().startswith("<result>") and chunk.strip().endswith("</result>")

def tokenize(text):
    return len(tokenizer.encode(text))

def test_fits_in_one_chunk():
    result = {
        "step_description": "Step 1",
        "tool_responses": [
            {"name": "toolA", "args": {}, "response": "short result"}
        ]
    }
    chunks = split_large_result(result, tokenizer, MAX_TOKENS)
    assert len(chunks) == 1
    assert tokenize(chunks[0]) <= MAX_TOKENS
    assert is_wrapped_in_result_tags(chunks[0])
    
def test_splits_into_multiple_chunks():
    result = {
        "step_description": "Step 1",
        "tool_responses": [
            {"name": f"tool{i}", "args": {}, "response": "x" * 300}
            for i in range(5)
        ]
    }
    chunks = split_large_result(result, tokenizer, MAX_TOKENS)
    assert len(chunks) > 1
    for chunk in chunks:
        assert tokenize(chunk) <= MAX_TOKENS
        assert is_wrapped_in_result_tags(chunk)
        
def test_single_oversized_response_in_own_chunk():
    result = {
        "step_description": "Step 1",
        "tool_responses": [
            {"name": "toolA", "args": {}, "response": "x" * 1500}
        ]
    }
    chunks = split_large_result(result, tokenizer, MAX_TOKENS)
    assert len(chunks) == 1
    assert tokenize(chunks[0]) > MAX_TOKENS  # Warning-worthy, but still returned
    assert is_wrapped_in_result_tags(chunks[0])
    
def test_large_step_description():
    result = {
        "step_description": "x" * 1000,
        "tool_responses": [
            {"name": "toolA", "args": {}, "response": "result A"},
            {"name": "toolB", "args": {}, "response": "result B"},
        ]
    }
    chunks = split_large_result(result, tokenizer, MAX_TOKENS)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert is_wrapped_in_result_tags(chunk)
        
def test_empty_tool_responses():
    result = {
        "step_description": "Just describing",
        "tool_responses": []
    }
    chunks = split_large_result(result, tokenizer, MAX_TOKENS)
    assert len(chunks) == 1
    assert is_wrapped_in_result_tags(chunks[0])
    
if __name__ == "__main__":
    test_fits_in_one_chunk()
    test_splits_into_multiple_chunks()
    test_single_oversized_response_in_own_chunk()
    test_large_step_description()
    test_empty_tool_responses()
