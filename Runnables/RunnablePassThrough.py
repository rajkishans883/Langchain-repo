from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")

# Initialize models
mistral_model = ChatMistralAI(
    api_key=mistral_key,
    model_name="mistral-tiny",
    temperature=0.3,        
)

hf_llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

hf_model = ChatHuggingFace(
    llm=hf_llm,
    max_tokens=1000,
)

# Define prompts
tweet_prompt = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"],
)

post_prompt = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"],
)

# Initialize parsers
parser = StrOutputParser()

# Example 1: Basic RunnablePassthrough
def basic_passthrough_example():
    """Demonstrates basic passthrough of input to output"""
    print("\n=== Basic RunnablePassthrough Example ===")

    # Create a chain that just passes through the input
    passthrough_chain = RunnablePassthrough()

    # The input will be passed through unchanged
    result = passthrough_chain.invoke({"topic": "AI ethics"})
    print("Passthrough result:", result)

# Example 2: Passthrough with modification
def modified_passthrough_example():
    """Demonstrates modifying the input while passing through"""
    print("\n=== Modified Passthrough Example ===")

    def add_prefix(input_dict):
        """Add a prefix to the topic"""
        input_dict["topic"] = f"Important: {input_dict['topic']}"
        return input_dict

    # Create a chain that modifies the input
    modified_chain = RunnablePassthrough() | add_prefix

    result = modified_chain.invoke({"topic": "AI ethics"})
    print("Modified passthrough result:", result)

# Example 3: Passthrough in parallel processing
def parallel_passthrough_example():
    """Demonstrates using passthrough in parallel chains"""
    print("\n=== Parallel Passthrough Example ===")

    # Create parallel chains with passthrough
    parallel_chain = RunnableParallel(
        original_topic=RunnablePassthrough(),  # Pass through original topic
        tweet_chain=RunnableSequence(tweet_prompt, mistral_model, parser),
        post_chain=RunnableSequence(post_prompt, hf_model, parser)
    )

    result = parallel_chain.invoke({"topic": "AI is dangerous or not"})
    print("Original topic:", result["original_topic"])
    print("Generated tweet:", result["tweet_chain"])
    print("Generated post:", result["post_chain"])

# Example 4: Passthrough with conditional branching
def conditional_passthrough_example():
    """Demonstrates conditional processing with passthrough"""
    print("\n=== Conditional Passthrough Example ===")

    def should_generate_post(input_dict):
        """Determine if we should generate a post based on topic length"""
        return len(input_dict["topic"].split()) > 3

    def select_chain(input_dict):
        """Select which chain to use based on condition"""
        if should_generate_post(input_dict):
            return RunnableSequence(post_prompt, hf_model, parser)
        return RunnablePassthrough()  # Just pass through if condition not met

    # Create a conditional chain
    conditional_chain = RunnablePassthrough() | select_chain

    # Test with short topic (will passthrough)
    short_result = conditional_chain.invoke({"topic": "AI good"})
    print("Short topic result:", short_result)

    # Test with long topic (will generate post)
    long_result = conditional_chain.invoke({"topic": "AI is changing our world rapidly"})
    print("Long topic result:", long_result)

# Example 5: Passthrough with multiple modifications
def complex_passthrough_example():
    """Demonstrates complex passthrough with multiple steps"""
    print("\n=== Complex Passthrough Example ===")

    def add_context(input_dict):
        """Add additional context to the input"""
        input_dict["context"] = "This is an important discussion about AI"
        return input_dict

    def format_input(input_dict):
        """Format the input for the model"""
        formatted = {
            "topic": input_dict["topic"],
            "additional_info": input_dict.get("context", "")
        }
        return formatted

    # Create a multi-step passthrough chain
    complex_chain = (
        RunnablePassthrough()
        | add_context
        | format_input
        | tweet_prompt
        | mistral_model
        | parser
    )

    result = complex_chain.invoke({"topic": "AI ethics"})
    print("Complex passthrough result:", result)

# Run all examples
if __name__ == "__main__":
    basic_passthrough_example()
    modified_passthrough_example()
    parallel_passthrough_example()
    conditional_passthrough_example()
    complex_passthrough_example()
