FunctionParameterPrompt = """As a Data Scientist, it's essential to provide comprehensive descriptions for each variable 
passed to a function. Ensure you write a summary for each variable that is used within the function, including variables
that are passed on to other functions.

For each variable, write a short 1 sentence summary that combines the answers to two questions; 
1. What is this variable? 2. What range of values can it take, if known?

For values on the `self` variable, additionally write an answer for all first-order attributes, i.e. `self.a` and `self.b`, 
but not `self.a.b`.

For each variable, specify its name and your answers. Only answer in the following format: VARIABLE_NAME: ANSWERS. 
Do not combine variables on one line. Ensure each variable is written on its own line.

When the variable resides in the self or cls object, use the notation self.variable or cls.variable and list it 
separately from the self variable. If variable does not reside in the self or cls object, represent it in its regular form. 

## Function
def my_function(self, X: float, parameter):
    assert X > 0 && X < 100
    assert isinstance(parameter, str)
    library_ability(self.item)
    y = inner_call(
        self.interesting_element, 
        self.other_element, 
        parameter * 2
    )
    self.combined = y * 2

## Summary
self: This is an instance of the class where my_function is called. Its attributes can vary based on the class definition.
self.interesting_element: This attribute is passed to inner_call, and its range and type depend on how it's defined in the class.
self.item: This attribute is passed to library_ability, and its range and type depend on how it's defined in the class.
self.other_element: Another attribute passed to inner_call, its range and type also depend on its class definition.
self.combined: An attribute calculated within my_function as twice the value returned by inner_call, it can take any value that can be obtained by multiplying a returned value of inner_call by 2.
X: This variable is a float type input data to my_function, constrained between 0 and 100 (exclusive).
parameter: This variable is a string type input to my_function and used in inner_call after being multiplied by 2.

## Function
{text}

## Summary
"""

FunctionRefinePrompt = """As a Data Scientist, it's essential to provide comprehensive descriptions for each variable 
passed to a function.

For each variable, write a short 1 sentence summary that combines the answers to two questions; 
1. What is this variable? 2. What range of values can it take, if known?

For values on the `self` variable, additionally write an answer for all first-order attributes, i.e. `self.a` and `self.b`, 
but not `self.a.b`.

For each variable, specify its name and your answers. Only answer in the following format: VARIABLE_NAME: ANSWERS. 
Do not combine variables on one line. Ensure each variable is written on its own line.

We have provided an existing answer up to a certain point.

Existing Answer:
{existing_answer}

Given the provided context, add and refine the previous answer. If the context doesn't change the answer, 
return the existing answer as it stands.
If you refine the answer, return the refined answer directly.

Context:
{text} 

Refined Answer:
"""
