import nim_nx
let a = tensor([2.0.float32, 1.0, 1.0, 2.0], f32).reshape(@[2, 2])
let b = tensor([5.0.float32, 4.0], f32)
echo "solve: ", solve(a, b)
echo "matrix_power: ", matrix_power(a, 2)
