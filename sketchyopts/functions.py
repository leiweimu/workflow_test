"""
This is the modules docstring.
"""

def very_important_function():
    """Very important function

    This function returns an array. 
    """
    j = [1,
       2,
       3
      ]
    return j

def foo():
    """
    This is another bogus function. 

    This function returns a string. 
    """
    print("All the newlines above me should be deleted!")

def bar(): 
    if True: 
      print("No newline above me!")
        
    print("There is a newline above me, and that's OK!")

def daily_average(temperatures: list[float]) -> float:
    """Computes daily average. 

    This computes $$\frac{1}{n} \sum_{i=1}^{n} t_i$$.

    References: 
        Zhuang et al, 2020: https://arxiv.org/abs/2010.07468

    Args:
        temperatures: list of daily temperatures

    Returns:
        average: average of daily temperature samples
        
    """
    return sum(temperatures) / len(temperatures)
