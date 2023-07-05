ComponentDescriberQAPrompt = """
You are a Staff Google programmer specialising in Machine Learning.

The user will be provide function source code or a python object. If the item is a function, provide an explanation of 
what each parameter to the function does. If the item is a python object, describe the object at a high level. 

Input:
{text}

Output:
"""


ComponentDescriberRefinePrompt = """
You are a Staff Google programmer specialising in Machine Learning.

The user will be provide function source code or a python object. If the item is a function, provide an explanation of 
what each parameter to the function does. If the item is a python object, describe the object at a high level. 

We have already started the summary below;
{existing_answer}

There is also this additional context;
{text} 

Given the additional context, refine the original summary. 
If the context isn't useful, return the existing summary.
"""
