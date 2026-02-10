import types, creation, backend, ops, shape
import std/math

proc fft*(tensor: Tensor): Tensor =
  let n = tensor.shape[0]
  var res = zeros(@[n], c64)
  let pSrc = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0])
  let pDst = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
  for k in 0..<n:
    var re: float32 = 0
    var im: float32 = 0
    for t in 0..<n:
      let angle = 2.0 * PI * k.float32 * t.float32 / n.float32
      re += pSrc[t] * cos(angle).float32
      im -= pSrc[t] * sin(angle).float32
    pDst[k * 2] = re
    pDst[k * 2 + 1] = im
  res
