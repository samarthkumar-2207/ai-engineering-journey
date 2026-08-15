def calculate_accuracy(correct : int, total : int) -> float:
  if total <= 0:
    raise ValueError("Total must be greater than zero.")
  if correct < 0:
    raise ValueError("Correct must be non-negative.")
  if correct > total:
    raise ValueError("Correct cannot be greater than total.")

  return correct / total

try:
  print(calculate_accuracy(80, 0))
except ValueError as e:
  print(f"Error: {e}")

  try:
    print(calculate_accuracy(-5, 100))
  except ValueError as e:
    print(f"Error: {e}")

try:
  print(calculate_accuracy(120, 100))
except ValueError as e:
  print(f"Error: {e}")

try:
  print(calculate_accuracy(80, 100))
except ValueError as e:
  print(f"Error: {e}")