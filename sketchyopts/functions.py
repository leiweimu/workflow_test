def very_important_function():
    """
    This is some documentation texts. 
    """
    j = [1,
       2,
       3
      ]
    return j

def foo():
    print("All the newlines above me should be deleted!")

def bar(): 
    if True: 
      print("No newline above me!")
        
    print("There is a newline above me, and that's OK!")

def daily_average(temperatures: list[float]) -> float:
    return sum(temperatures) / len(temperatures)
