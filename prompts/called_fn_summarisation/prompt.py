CalledFunctionQuestionPrompt = """
You are a Data Scientist producing a summary of a function call. Given the existing description below, write a summary 
of the function: {text}

Summary:"""


CalledFunctionRefinePrompt = """
You are a Data Scientist producing a final summary of a function call. We have provided an existing summary up to a certain point: {existing_answer}\n
We have the opportunity to refine the existing summary (only if needed) with some more context below.

{text} 

Given the new context, refine the original summary. 
If the context isn't useful, return the existing summary.
"""
