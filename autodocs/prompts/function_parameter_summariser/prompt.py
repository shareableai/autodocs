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

Given the provided context, add and refine the previous answer. If the context doesn't change the answer, 
return the existing answer as it stands.
If you refine the answer, return the refined answer directly.

Context:
{text} 

Summary:
"""
