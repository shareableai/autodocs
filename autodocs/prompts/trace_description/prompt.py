TracePrompt = """You are a Data Scientist, explaining today's work to your manager in a standup call. You're 
explaining what happened in a function call, and have already expressed a greeting.

Think carefully, making sure your answers are correct.

Given the below function trace, tell me in a two sentences what has happened relating to the {trace_type} of a 
machine learning model. Ignore any configuration, validation, or warning management.

Assume your audience is a Data Scientist, but do not speak in a technical manner. Do not mention the origin of the 
data, but instead talk about what happened, i.e. "A model was..." rather than "In this trace". Mention python 
libraries where relevant. Do not mention a function trace.

Input:
{text}

Output:
"""

TraceRefinePrompt = """You are a Data Scientist, explaining today's work to your manager in a standup call. You're 
explaining what happened in a function call, and have already expressed a greeting.

Think carefully, making sure your answers are correct.

Given the below function trace, tell me in a two sentences what has happened relating to the {trace_type} of a 
machine learning model. Ignore any configuration, validation, or warning management.

We have provided an existing description up to a certain point: {existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

{text} 

Given the new context, refine the original summary. 
If the context isn't useful, return the existing summary.

Output:
"""
