import nim_nx
let k = key(42)
let (v, new_key) = uniform(k, @[2], dtype=f32)
echo "uniform: ", v
