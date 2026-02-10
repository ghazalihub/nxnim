import std/macros

macro defn*(body: untyped): untyped =
  # Simple macro that just returns the body for now
  # In a more complete implementation, this would transform the code to build a static graph
  body
