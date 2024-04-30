
from nemoguardrails import LLMRails, RailsConfig

# Load a guardrails configuration from the specified path.
config = RailsConfig.from_path("config")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": "Hello world!"}]
)

print(response["content"])

response = rails.generate(messages=[{
    "role": "user",
    "content": "Tell me about Modi ji?"
}])
print(response["content"])

