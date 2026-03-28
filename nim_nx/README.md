# Nim Nx

A Nim clone of the Elixir Nx library.

## Features

- **Tensors**: N-dimensional arrays with support for multiple dtypes (`f32`, `c64`, `u8`, etc).
- **Broadcasting**: Automatic shape broadcasting for element-wise operations.
- **Autograd**: Reverse-mode automatic differentiation with support for complex graphs.
- **FFT**: Fast Fourier Transform (1D and 2D).
- **Convolution**: Generic N-D convolution with `VALID` and `SAME` padding.
- **Random**: JAX-style PRNG system with `split`, `uniform`, and `normal`.
- **Linear Algebra**: Basic matrix operations and inversion.
- **Constants**: Mathematical constants like `pi`, `e`, etc.

## Usage

```nim
import nim_nx

let x = tensor(@[1.0f32, 2.0, 3.0]).with_grad()
let y = multiply(x, x)
let z = sum(y)

z.backward()

echo x.grad # [2.0, 4.0, 6.0]
```
