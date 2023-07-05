CalledFunctionParameterPrompt = """
Your task is to write a condensed technical description of a function, focusing on the parameters provided to the function. For each parameter, provide an
explanation of the impact that parameter has on the function.

{text}
"""

CalledFunctionQuestionPrompt = """
Your task is to write a condensed technical description of a function.

Using the Function Source and Function Arguments, write a description of what the function does, focusing on components that help explain the transformations the model applies to the input data. 

Function Source:
def multiply(self, y: int) -> int:
    result = int(self.x * y)
    if self.must_be_positive and result < 0:
        raise ValueError
    return result

Function Arguments:
self = MathsOperator(x=2), y = 5

Description:
Multiply is a method of the class MathsOperator that returns the integer coercion of self.x * y. If self.must_be_positive is True, the function asserts that the result is positive.

Function Source:
def is_pii(self, X: str) -> bool:
    for word in self.people:
        if self.similar(X,word) > self.threshold:
            return True
    return False

Function Arguments:
self = WordClassifier(people=[david, richard], threshold=0.5), X = 'simon'

Description:
is_pii is a method of the class WordClassifier that returns a boolean expressing if the string X is pii. This is determined based on whether X is similar to any word in the class variable self.people. 
Similarity is identified based on the result of the self.similar class function which takes two str-like objects and returns a float. The float is compared to self.threshold to determine
if the words are similar enough. 

{text}

Description:
"""


CalledFunctionRefinePrompt = """
You are a Data Scientist producing the final summary of function code for a technical colleague.

Using the Function Source and Function Arguments, write a description of what the function does, focusing on components that help explain the transformations the model applies to the input data. 

Function Source:
def multiply(self, y: int) -> int:
    result = int(self.x * y)
    if self.must_be_positive and result < 0:
        raise ValueError
    return result

Function Arguments:
self = MathsOperator(x = 2), y = 5

Description:
Multiply is a method of the class MathsOperator that returns the integer coercion of self.x * y. If self.must_be_positive is True, the function asserts that the result is positive.

Function Source:
def is_pii(self, X: str) -> bool:
    for word in self.people:
        if self.similar(X,word) > self.threshold:
            return True
    return False

Function Arguments:
self = WordClassifier(people=[david, richard]), X = 'simon'

Description:
is_pii is method of the class WordClassifier that returns a boolean expressing if the string X is pii. This is determined based on whether X is similar to any word in the variable self.people. 
Similarity is identified based on the result of the `self.similar` function which takes two str-like objects and returns a float. The float is compared to self.threshold to determine
if the words are similar enough. 


We have provided an existing description up to a certain point: {existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

{text} 

Given the new context, refine the original summary. 
If the context isn't useful, return the existing summary.
"""
