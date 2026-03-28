import nim_nx
let a = tensor([1.0.float32, 2.0], f32)
let b = tensor([3.0.float32, 4.0], f32)
echo "add: ", a + b
echo "multiply: ", a * b
echo "exp: ", exp(a)
