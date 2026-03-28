import nim_nx
let t = tensor([1.0.float32, 2.0, 3.0, 4.0], f32)
echo "cumulative_product: ", t.cumulative_product()
echo "cumulative_max: ", t.cumulative_max()
echo "argmin: ", t.argmin()
