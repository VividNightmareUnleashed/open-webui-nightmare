# Writer Assistant Tool

Enables any model to delegate writing tasks to another specialized model via OpenRouter. This allows the primary model to request help with content generation, editing, rewriting, or any other writing-related tasks.

## Description

The Writer Assistant tool provides a **model-to-model delegation** capability where the primary assistant can call out to a specialized writing model (via OpenRouter) for tasks like:
- Drafting emails, letters, or documents
- Rewriting content in different styles
- Expanding or condensing text
- Translating or paraphrasing
- Creative writing tasks
- Technical writing and documentation

This is particularly useful when you want to:
- Use a fast reasoning model for conversation but delegate writing to a more creative model
- Offload time-consuming writing tasks to a specialized model
- Maintain conversation flow while generating high-quality written content

## Features

- **Simple Interface** - Model provides only one parameter: a prompt
- **Model-to-Model Delegation** - Primary model can invoke another model for writing tasks
- **OpenRouter Integration** - Uses OpenRouter API for access to multiple models
- **Configurable** - Choose any OpenRouter-supported model via Valves
- **Error Handling** - Graceful error messages if API calls fail
- **Debug Mode** - Optional logging for troubleshooting

## Installation

1. Copy the contents of [writer_assistant.py](writer_assistant.py)
2. In Open WebUI, go to **Workspace → Tools**
3. Click **Create New Tool**
4. Paste the code
5. Save and enable the tool

## Configuration

### Required Setup

**You must configure your OpenRouter API key** before using this tool.

#### Option 1: Set via Environment Variable (Recommended)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

#### Option 2: Set in Tool Valves
1. Go to **Workspace → Tools**
2. Find "Writer Assistant" and click settings
3. Set `OPENROUTER_API_KEY` to your API key

### Optional Configuration

Configure these in the tool's Valves settings:

- **TARGET_MODEL**: Default model for writing tasks (default: `anthropic/claude-3.5-sonnet`)
  - Examples: `openai/gpt-4o`, `google/gemini-pro-1.5`, `meta-llama/llama-3.1-70b-instruct`
  - See [OpenRouter models](https://openrouter.ai/models) for full list

- **SYSTEM_PROMPT**: Instructions for the writer model (default: "You are a professional writing assistant...")

- **MAX_TOKENS**: Maximum response length (default: 4000)

- **TEMPERATURE**: Creativity level (default: 0.7, range: 0.0-2.0)

- **DEBUG**: Enable debug logging (default: false)

## Usage

The model calls the tool with a single parameter: **`prompt`**

### Basic Example

```python
# The model invokes the tool like this:
result = write_content(
    prompt="Write a professional email thanking a client for their business"
)
```

### Content Rewriting

```python
result = write_content(
    prompt="Rewrite this to be more concise: I wanted to reach out to you today to let you know that we really appreciate your business and all the work you've done with us over the past year..."
)
```

### With Style Specification

```python
result = write_content(
    prompt="Write a product announcement in a casual and friendly style for our new mobile app"
)
```

### Creative Writing

```python
result = write_content(
    prompt="Write a creative short story about a time traveler who gets stuck in ancient Rome"
)
```

### Technical Documentation

```python
result = write_content(
    prompt="Write clear API documentation for this function: def calculate_discount(price, percentage): return price * (1 - percentage/100)"
)
```

### Email with Context

```python
result = write_content(
    prompt="Write a follow-up email after a job interview. Context: Company: TechCorp, Position: Senior Developer, Interviewer: Sarah Chen. Tone: professional and enthusiastic."
)
```

## Example Conversation

**User**: "Can you draft a thank you email for my team meeting?"

**Assistant** (thinking): "I'll delegate this writing task to the writer assistant."

**Assistant** (calls tool):
```python
write_content(
    prompt="Write a professional thank you email for a productive team meeting. Tone should be warm and appreciative."
)
```

**Tool returns**: "Subject: Thank You for a Great Meeting..."

**Assistant**: "Here's a draft email for you:

Subject: Thank You for a Great Meeting

[... content from writer model ...]

Would you like me to adjust anything?"

## Use Cases

### Email Drafting
```python
write_content(
    prompt="Write a professional email requesting a meeting to discuss Q1 budget planning"
)
```

### Content Expansion
```python
write_content(
    prompt="Expand this outline into a full blog post: 1. Introduction to AI, 2. Benefits, 3. Challenges, 4. Future outlook. Style: technical but accessible."
)
```

### Summarization
```python
write_content(
    prompt="Condense this article into 3 key bullet points: [long article text]"
)
```

### Translation/Paraphrasing
```python
write_content(
    prompt="Rephrase this technical explanation in simple language for a 10-year-old: [complex text]"
)
```

## Model Recommendations

Different models excel at different writing tasks. Configure via `TARGET_MODEL` valve:

- **General Purpose Writing**: `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`
- **Creative Writing**: `anthropic/claude-3-opus`, `google/gemini-pro-1.5`
- **Technical Writing**: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`
- **Cost-Effective**: `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`
- **Long-Form Content**: Models with high context windows

See [OpenRouter models](https://openrouter.ai/models) for detailed comparisons.

## Technical Details

### How It Works

1. **Tool Invocation**: Primary model calls `write_content(prompt="...")`
2. **Request Building**: Tool constructs an OpenRouter API request with:
   - System prompt (configured in valves)
   - User message (the prompt)
   - Model selection (from valves)
   - Generation parameters (from valves)
3. **API Call**: Makes HTTP POST to OpenRouter's `/chat/completions` endpoint
4. **Response Processing**: Extracts content from API response
5. **Return**: Returns written content as string to primary model

### API Integration

The tool uses OpenRouter's OpenAI-compatible API:
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Format: OpenAI Chat Completions format
- Authentication: Bearer token via `Authorization` header

### Error Handling

The tool handles various error conditions:
- Missing API key → Returns error message
- Network errors → Returns error with details
- API errors (4xx/5xx) → Returns status code and error text
- Malformed responses → Returns parsing error
- Empty responses → Returns "no content" error

All errors are logged for debugging.

## Troubleshooting

### "OPENROUTER_API_KEY is not set"
- Set the API key via environment variable or tool valves
- Get your API key from [OpenRouter](https://openrouter.ai/)

### "OpenRouter API error (status 401)"
- Invalid API key
- API key not properly formatted (should start with `sk-or-v1-`)

### "OpenRouter API error (status 402)"
- Insufficient credits in your OpenRouter account
- Add credits at [OpenRouter account page](https://openrouter.ai/account)

### "Network error calling OpenRouter"
- Check internet connectivity
- Verify OpenRouter is not blocked by firewall
- Check if OpenRouter service is operational

### Tool not appearing in model's available tools
- Ensure tool is enabled in Open WebUI
- Check that tool is assigned to the model you're using
- Try refreshing the chat or creating a new chat

### Enable Debug Mode
Set `DEBUG: true` in tool valves to see detailed logs of:
- Prompt (first 100 characters)
- Target model
- API request body
- API response data
- Errors and exceptions

## Limitations

- **Single Parameter**: Model provides only a `prompt` string (simplicity vs flexibility trade-off)
- **No Streaming**: Tool waits for complete response (not streaming)
- **Single Turn**: Each call is independent (no conversation context between calls)
- **No Nested Calls**: Writer model cannot call tools itself
- **API Costs**: Each invocation costs credits on OpenRouter
- **Rate Limits**: Subject to OpenRouter's rate limits

## Cost Considerations

- Each tool invocation = 1 API call to OpenRouter
- Costs vary by model (see [OpenRouter pricing](https://openrouter.ai/models))
- Configure `MAX_TOKENS` to limit response length and costs
- Use cost-effective models for simple tasks
- Reserve expensive models for complex writing

## Security Notes

- **API Key Security**: Never commit API keys to version control
- **Environment Variables**: Prefer environment variables over valves for sensitive data
- **User Input**: Prompt comes from model, not directly from user
- **Error Messages**: Tool avoids exposing sensitive information in error messages

## FAQ

**Q: Can I use different models for different tasks?**
A: No, the model is configured in Valves. You'd need to create multiple instances of the tool or change the Valves configuration.

**Q: Can the prompt include the content to rewrite?**
A: Yes! Just include everything in the prompt string, e.g., `"Rewrite this to be shorter: [your content]"`

**Q: Does this support streaming?**
A: No, the tool waits for the complete response before returning.

**Q: Can I control temperature or max_tokens per call?**
A: No, these are configured in the Valves and apply to all calls.

**Q: What if I want the writer to use a different style?**
A: Include the style instruction in your prompt, e.g., `"Write this in a casual, friendly tone: ..."`

## Future Enhancements

Potential improvements for future versions:
- Optional parameters for per-call overrides (temperature, max_tokens, model)
- Streaming support for real-time output
- Conversation context for multi-turn writing tasks
- Cost tracking and usage statistics
- Multiple model attempts with fallback
- Caching for repeated requests

## Related Tools

- **Pro Mode Action** - Switches entire conversation to a different model
- **Extended Thinking Filter** - Adds reasoning effort to model responses
- **Verbose Filter** - Controls output verbosity

## Contributing

Contributions are welcome! If you have ideas for improvements:
1. Open an issue on GitHub
2. Submit a pull request
3. Share your use cases and model recommendations

## License

MIT License - see repository LICENSE file

---

**Need Help?** Open an issue at [GitHub repository](https://github.com/jrkropp/open-webui-developer-toolkit/issues)
