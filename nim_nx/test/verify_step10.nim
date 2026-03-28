import nim_nx
let t = tensor([1.0.float32, 0.0, 0.0, 0.0], f32)
echo "fft: ", fft(t)
