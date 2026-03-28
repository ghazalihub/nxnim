import nim_nx

let pred = tensor([1u8, 0, 1], u8)
let on_true = tensor([10.0.float32, 10.0, 10.0], f32)
let on_false = tensor([0.0.float32, 0.0, 0.0], f32)
echo "select: ", select(pred, on_true, on_false) # Expected [10, 0, 10]

let init = tensor([0.0.float32], f32)
let loop_res = while_loop(
  proc(t: Tensor): Tensor = less(t, tensor([5.0.float32], f32)),
  proc(t: Tensor): Tensor = t + tensor([1.0.float32], f32),
  init
)
echo "while_loop: ", loop_res # Expected 5.0

let t_scan = tensor([1.0.float32, 2.0, 3.0], f32)
let (final_acc, scanned) = scan(
  tensor([0.0.float32], f32),
  t_scan,
  proc(acc: Tensor, x: Tensor): (Tensor, Tensor) = (acc + x, acc + x)
)
echo "scan final: ", final_acc # Expected 6.0
echo "scan results: ", scanned # Expected [1, 3, 6]
