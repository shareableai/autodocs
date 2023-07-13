TracePrompt = """The below text is a function trace. For each line, identify what the function does, and whether the 
function relates to configuration, validation, or something else.

Create a list of the functions that exclusively relate to the {trace_type} of a model, and do not relate to 
configuration, warnings, or checks.

Of the functions in the list you've just created, draw them in an indented tree graph to show hierarchy, using a tree 
glyph. Input: {text}

Output:
"""

TraceRefinePrompt = """The below text is a function trace. For each line, identify what the function does, 
and whether the function relates to configuration, validation, or something else.

Create a list of the functions that exclusively relate to the {trace_type} of a model, and do not relate to configuration, 
warnings, or checks.

Of the functions in the list you've just created, draw them in an indented tree graph to show hierarchy, using a tree 
glyph.

We have provided an existing description up to a certain point: {existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

{text} 

Given the new context, refine the original summary. 
If the context isn't useful, return the existing summary.

Output:
"""
