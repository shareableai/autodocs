CondenserPrompt = """
You are a Data Scientist describing the transformations that are applied to input data by a machine learning
model during the {trace_type} stage.

Given the following ordered function descriptions, describe each transformation that happens to the input data, keeping a 
focus on the input data at each step. 

{text}

Steps:
"""

CondenserRefinerPrompt = """
You are a Data Scientist describing the transformations that are applied to input data by a machine learning
model during the {trace_type} stage.

We have provided an existing set of steps up to a certain point: {existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below.

{text}

Given the new context, refine the original summary. 
If the new context isn't useful, return the original summary.
"""
