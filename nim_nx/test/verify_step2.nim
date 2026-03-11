import nim_nx
let a = tensor([1.0.float32, 2.0, 3.0, 4.0], f32).reshape(@[2, 2])
echo "sum: ", a.sum()
echo "mean: ", a.mean()
echo "sum(axes=@[0]): ", a.sum(axes= @[0])
