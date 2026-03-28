import nim_nx
let a = tensor([1.0.float32, 2.0], f32)
let b = tensor([3.0.float32, 4.0, 5.0], f32)
echo "outer: ", outer(a, b)
echo "transpose: ", tensor([1.0.float32, 2.0, 3.0, 4.0], f32).reshape(@[2, 2]).transpose()
