ComponentIdentifierQAPrompt = """
You are a Data Scientist identifying the Machine Learning Components within a set of functions for a ML {trace_type} workflow.

Given the functions provided, and their arguments, describe the ML components in one to two sentences.


Provide the components in the following format;
# Components
(Component Name): Component Description
(Component Name): Component Description
...


Functions;
{text}

# Components
"""


ComponentsRefinePrompt = """
You are a Data Scientist identifying the Machine Learning Components within a set of functions for a ML {trace_type} workflow.

Given the functions provided, and their arguments, describe the ML components in one to two sentences. Focus on the general purpose of
the component, rather than specifics.

We have provided an existing set of components up to a certain point:
{existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

{text} 

Given the new context, refine the original summary. 
If the context isn't useful, return the existing summary.
"""
