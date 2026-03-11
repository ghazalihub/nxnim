import nim_nx
let x = tensor([2.0.float32, 3.0], f32).with_grad()
let g = grad(proc(t: Tensor): Tensor = t * t + t)
echo "grad(x^2 + x): ", g(x) # Expected [5, 7]
