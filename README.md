# Automatically Generate Model Documentation

## Getting Started

### Environment Variables
* OPENAI_API_KEY - Supply this from your OpenAI Account.

### Setup
1. Using [Observer](https://www.github.com/shareableai/observer), annotate your model's runtime. 
2. Find the appropriate TraceID from your `.stack_traces` directory
3. Create Documentation by running autodocs using one of your trace ids.

```bash
python autodocs/__main__.py {trace_id}
```