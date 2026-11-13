# Common Pitfalls and Guidelines in SWE Bug Fixing

1. **Avoid Over-Fixing**: Fix only the specific bug described in the issue. Modifying other parts of the codebase increases regression risk.
2. **Whitespace and Line Endings**: Keep existing indentation style exactly as found in `read_file`.
3. **Callable & None Guards**: Common python bug types include:
   - Calling an object that can be either a function or a string/value (use `callable(obj)` guard).
   - Division by zero or empty collection indexing (check bounds or length first).
   - Mutable default arguments in function definitions.
4. **Final Patch Submission**: After verifying tests with `run_command`, always execute `submit_patch()` to complete the issue.
