from ddgs import ddgs
from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> str:
  """Adds two numbers together (a + b) and returns the formatted result."""
  result = a + b
  return f"{a} + {b} = {result}"


@tool
def subtract(a: float, b: float) -> str:
  """Subtracts the second number from the first (a - b) and returns the formatted result."""
  result = a - b
  return f"{a} - {b} = {result}"


@tool
def multiply(a: float, b: float) -> str:
  """Multiplies two numbers together (a * b) and returns the formatted result."""
  result = a * b
  return f"{a} * {b} = {result}"


@tool
def divide(a: float, b: float) -> str:
  """Divides the first number by the second number (a / b) and returns the formatted result.

  Returns an error message if division by zero is attempted.
  """
  if b == 0:
    return "Error: Division by zero is not allowed."
  result = a / b
  return f"{a} / {b} = {result}"


@tool
def power(base: float, exponent: float) -> str:
  """Raises the base to the power of the exponent (base ** exponent) and returns the formatted result."""
  result = base**exponent
  return f"{base} ** {exponent} = {result}"


@tool
def search(query: str) -> str:
  """Search the internet for real-time information.

  Use this for news, stocks, weather, or current events.

  Args:
      query: The search term or phrase.
  """
  try:
    with ddgs() as client:
      results = [r for r in client.text(query, max_results=3)]

    if not results:
      return f"Not found anything about this query: {query}"

    formatted_results = []
    for r in results:
      formatted_results.append(
          f"Text: {r.get('title')}\nMain Data: {r.get('body')}\nURL = {r.get('href')}"
      )

    return "\n-------\n".join(formatted_results)
  except Exception as error:
    return f"Search got an error:\n{error}"


# LangChain Agent ke liye saare tools ki ek combined list
agent_tools = [add, subtract, multiply, divide, power, search]