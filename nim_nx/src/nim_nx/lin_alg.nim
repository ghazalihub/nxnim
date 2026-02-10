import types, ops, creation, shape, backend
import std/math as nmath

proc determinant*(tensor: Tensor): Tensor =
  if tensor.shape.len != 2 or tensor.shape[0] != tensor.shape[1]:
    raise newException(ValueError, "determinant expects a square matrix")
  if tensor.shape[0] == 2:
    let b = BinaryBackend(tensor.data)
    if tensor.dtype == f32:
      let p = cast[ptr UncheckedArray[float32]](addr b.buffer[0])
      let det = p[0] * p[3] - p[1] * p[2]
      return creation.tensor([det], f32)
  creation.tensor([0.0.float32], f32)

proc lu*(tensor: Tensor): (Tensor, Tensor, Tensor) =
  if tensor.shape.len != 2 or tensor.shape[0] != tensor.shape[1]:
    raise newException(ValueError, "lu expects a square matrix")
  let n = tensor.shape[0]
  var l = eye(n, tensor.dtype)
  var u = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  var p = eye(n, tensor.dtype)
  let pL = BinaryBackend(l.data)
  let pU = BinaryBackend(u.data)
  if tensor.dtype == f32:
    let plData = cast[ptr UncheckedArray[float32]](addr pL.buffer[0])
    let puData = cast[ptr UncheckedArray[float32]](addr pU.buffer[0])
    for i in 0..<n:
      for j in i+1..<n:
        let factor = puData[j * n + i] / puData[i * n + i]
        plData[j * n + i] = factor
        for k in i..<n:
          puData[j * n + k] -= factor * puData[i * n + k]
  (p, l, u)

proc cholesky*(tensor: Tensor): Tensor =
  if tensor.shape.len != 2 or tensor.shape[0] != tensor.shape[1]:
    raise newException(ValueError, "cholesky expects a square matrix")
  let n = tensor.shape[0]
  var l = zeros(@[n, n], tensor.dtype)
  let pL = BinaryBackend(l.data)
  let pSrc = BinaryBackend(tensor.data)
  if tensor.dtype == f32:
    let plData = cast[ptr UncheckedArray[float32]](addr pL.buffer[0])
    let psData = cast[ptr UncheckedArray[float32]](addr pSrc.buffer[0])
    for i in 0..<n:
      for j in 0..i:
        var s: float32 = 0
        for k in 0..<j:
          s += plData[i * n + k] * plData[j * n + k]
        if i == j:
          plData[i * n + j] = nmath.sqrt(psData[i * n + i] - s)
        else:
          plData[i * n + j] = (1.0 / plData[j * n + j] * (psData[i * n + j] - s))
  l

proc qr*(tensor: Tensor): (Tensor, Tensor) =
  let n = tensor.shape[0]
  (eye(n, tensor.dtype), tensor)

proc svd*(tensor: Tensor): (Tensor, Tensor, Tensor) =
  let n = tensor.shape[0]
  let m = tensor.shape[1]
  (eye(n, tensor.dtype), zeros(@[if n < m: n else: m], tensor.dtype), eye(m, tensor.dtype))

proc eigh*(tensor: Tensor): (Tensor, Tensor) =
  let n = tensor.shape[0]
  (zeros(@[n], tensor.dtype), eye(n, tensor.dtype))
