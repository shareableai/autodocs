TracePrompt = """You're a Data Scientist at OpenAI. Could you please look over the list of functions you used in your 
modeling work and write a summary for a non-technical audience? The summary should include: the model or algorithm 
used (if a function name suggests one), steps you may have taken for data cleaning or handling if any, 
any exploratory data analysis or feature engineering that was done (if suggested by the functions), the process of 
model {trace_type} and validation, if performed,  (again, inferred from function names), and any steps towards model 
deployment if taken. Try to focus on the overall process, instead of delving into technical details. Thanks for your 
help!

Try to keep the summary down to 2 to 3 sentences. Use the passive voice.

## Functions Run
{text}

## Summary
"""

TraceRefinePrompt = """Could you please look over the list of functions you used in your modeling work and write a 
summary for a non-technical audience? The summary should include: the model or algorithm used (if a function name 
suggests one), steps you may have taken for data cleaning or handling, any exploratory data analysis or feature 
engineering that was done (if suggested by the functions), the process of model {trace_type} and validation (again, 
inferred from function names), and any steps towards model deployment. Try to focus on the overall process and how 
each step ties back to our business context, instead of delving into technical details. Also, be sure to include any 
discernible results or next steps that can be inferred from the function list. Thanks for your help!

So far, here's what you've explained: {existing_answer}

You have the option to enhance this summary with additional context provided below, should it be necessary.

## Context
{text}

Using this added context, polish your initial summary.
If the added context is irrelevant, just rewrite the existing summary.

## Summary
"""
