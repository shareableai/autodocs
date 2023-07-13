FunctionParameterPrompt = """You are a Data Scientist writing a summary for each parameter passed to a function.

Think carefully, making sure your answers are correct.

For each parameter, write a short paragraph that combines the answers to two questions; 1. What is this parameter? 2. 
What range of arguments can it accept, if known?

For each parameter, write the parameter name and your paragraph. If a parameter is on the `self` or `cls` object, 
write it as `self.parameter` or `cls.parameter` on a separate line to the self parameter. Otherwise, write it out 
normally.

Input:
{text}

Summary:
"""

FunctionRefinePrompt = """You are a Data Scientist writing a summary for each parameter passed to a function.

Think carefully, making sure your answers are correct.

For each parameter, write a short paragraph that combines the answers to two questions; 1. What is this parameter? 2. 
What range of arguments can it accept, if known?

For each parameter, write the parameter name and your paragraph. If a parameter is on the `self` or `cls` object, 
write it as `self.parameter` or `cls.parameter` on a separate line to the self parameter. Otherwise, write it out 
normally.


We have provided an existing summary up to a certain point.

Existing Summary:
{existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

Context:
{text} 

Given the new context, refine the original description. 
If the context isn't useful, return the existing description.
If no refinements are necessary, return the existing description.

Summary:
"""
