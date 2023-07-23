import pathlib

from autodocs.prompts.function_parameter_summariser.run import FnParameterSummarizer
from autodocs.utils.function_description import FunctionDescription

INPUT_TEXT = """
def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):\n    if start_token is None:\n        assert context is not None, 'Specify exactly one of start_token and context!'\n        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)\n    else:\n        assert context is None, 'Specify exactly one of start_token and context!'\n        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)\n    prev = context\n    output = context\n    past = None\n    with torch.no_grad():\n        for i in trange(length):\n            logits, past = model(prev, past=past)\n            logits = logits[:, -1, :] / temperature\n            logits = top_k_logits(logits, k=top_k)\n            log_probs = F.softmax(logits, dim=-1)\n            if sample:\n                prev = torch.multinomial(log_probs, num_samples=1)\n            else:\n                _, prev = torch.topk(log_probs, k=1, dim=-1)\n            output = torch.cat((output, prev), dim=1)\n            Observer.pause()\n    return output\n
"""


def test_summary():
    desc = FnParameterSummarizer()(FunctionDescription(
        name="Example",
        source=INPUT_TEXT,
        docs="",
        arguments={},
        root_dir=pathlib.Path.cwd(),
        caller_docs=None,
        caller_name=None,
        signature=None,
        tracked_argument_ids={}
    ))
    print(desc)
    breakpoint()

