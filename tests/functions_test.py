import pytest

from sketchyopts.functions import *

def test_very_important_function(): 
  assert very_important_function() == [1,2,3]

def test_foo(capsys): 
  foo()
  captured = capsys.readouterr()
  assert captured.out == "All the newlines above me should be deleted!"
