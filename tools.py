from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
  """Adds two numbers together (a + b) and returns the result."""
  return a + b


@tool
def subtract(a: float, b: float) -> float:
  """Subtracts the second number from the first (a - b) and returns the result."""
  return a - b


@tool
def multiply(a: float, b: float) -> float:
  """Multiplies two numbers together (a * b) and returns the result."""
  return a * b


@tool
def divide(a: float, b: float) -> float:
  """Divides the first number by the second number (a / b).

  Returns an error message if division by zero is attempted.
  """
  if b == 0:
    return "Error: Division by zero is not allowed."
  return a / b


@tool
def power(base: float, exponent: float) -> float:
  """Raises the base to the power of the exponent (base ** exponent) and returns the result."""
  return base**exponent


# Saare tools ki ek list jise aap LangChain Agent me pass kar sakte hain:
arithmetic_tools = [add, subtract, multiply, divide, power]