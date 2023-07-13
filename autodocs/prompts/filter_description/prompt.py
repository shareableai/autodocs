FilterPrompt = """
Your task is to filter an input to {condition}, with an output format of {output_format}. 

{extra_conditions}

Input:
{text}

Output:
"""

FilterRefinePrompt = """
Your task is to filter an input to {condition}, with an output format of {output_format}.

{extra_conditions}

Some of the input has already been filtered to: {existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

{text} 

Given the new context, refine the original summary. 
If the context isn't useful, return the existing summary.
"""
