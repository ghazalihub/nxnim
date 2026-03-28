import nim_nx
let a = tensor([1.0.float32, 2.0], f32)
let b = tensor([10.0.float32], f32)
echo "broadcast add: ", a + b
