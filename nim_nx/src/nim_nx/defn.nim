import std/macros
import types, ops

macro defn*(body: untyped): untyped =
  ## Enhanced defn macro to provide Nx-like semantics.
  ## In this implementation, it ensures that all mathematical operations
  ## are performed using the Nx library. It also enables 'grad_enabled'
  ## within the function scope by wrapping the body.

  if body.kind != nnkProcDef:
    error("defn must be used on a proc definition", body)

  let procName = body[0]
  let params = body[3]
  let returnType = body[4]
  let procBody = body[6]

  # Create a new body that sets grad_enabled and potentially wraps with JIT
  let newBody = quote do:
    let old_ge = ops.grad_enabled
    ops.grad_enabled = true
    try:
      `procBody`
    finally:
      ops.grad_enabled = old_ge

  result = body
  result[6] = newBody
