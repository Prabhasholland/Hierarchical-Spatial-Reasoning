# ARC-AGI-2 Official Submission Format

Based on standard ARC Prize Kaggle competitions (ARC Prize 2024 / ARC-AGI-2):

## 1. Test Task Input Format
Test tasks consist of `train` pairs (input, output) and `test` pairs (input only). Some tasks have 1 test pair, some have 2. 

## 2. submission.json Format
A dictionary where each key is a `task_id` (string), and the value is a list of prediction objects, one for each test pair in the task.
Each prediction object contains exactly two keys: `attempt_1` and `attempt_2`.
The value for each attempt is a 2D array of integers (list of lists).

Example:
```json
{
  "00576224": [
    {
      "attempt_1": [[0, 1], [2, 3]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ],
  "00d62c1b": [
    {
      "attempt_1": [[1, 2, 3]],
      "attempt_2": [[1, 2, 3]]
    },
    {
      "attempt_1": [[4, 5, 6]],
      "attempt_2": [[4, 5, 6]]
    }
  ]
}
```

## 3. Allowed Values & Dimensions
- **Grid dimensions**: Must be between 1x1 and 30x30.
- **Allowed values**: Integers from 0 to 9 inclusive.
- No strings, no floats.

## 4. Fallback Rule
If the solver fails to produce a prediction, or crashes, it MUST output a valid default grid (e.g., `[[0, 0], [0, 0]]` or just echo the input) for BOTH attempts. Failure to provide exactly `attempt_1` and `attempt_2` for every test pair of every test task will result in a Kaggle Submission Error.

## 5. Notebook Restrictions
- Network/Internet: OFF
- Runtime limit: Usually 12 hours total.
- Submissions are made by writing exactly to `/kaggle/working/submission.json`.
