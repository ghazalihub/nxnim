import types, ops, creation, shape, backend
import std/math as nmath

template luImpl(T: typedesc) =
  for i in 0..<n:
    var maxIdx = i
    var maxVal = system.abs(puData[i * n + i])
    for j in i + 1 ..< n:
      if system.abs(puData[j * n + i]) > maxVal:
        maxVal = system.abs(puData[j * n + i]); maxIdx = j
    if maxIdx != i:
      swaps += 1
      for k in 0..<n:
        let temp = puData[i * n + k]; puData[i * n + k] = puData[maxIdx * n + k]; puData[maxIdx * n + k] = temp
        let tempP = ppData[i * n + k]; ppData[i * n + k] = ppData[maxIdx * n + k]; ppData[maxIdx * n + k] = tempP
      for k in 0..<i:
        let tempL = plData[i * n + k]; plData[i * n + k] = plData[maxIdx * n + k]; plData[maxIdx * n + k] = tempL
    let diagVal = puData[i * n + i]
    if system.abs(diagVal) > 1e-12:
      for j in i+1..<n:
        let factor = puData[j * n + i] / diagVal; plData[j * n + i] = factor
        for k in i..<n: puData[j * n + k] -= factor * puData[i * n + k]

proc lu_internal(tensor: Tensor): (Tensor, Tensor, Tensor, int) =
  if tensor.shape.len != 2 or tensor.shape[0] != tensor.shape[1]: raise newException(ValueError, "lu expects square matrix")
  let n = tensor.shape[0]; var l = eye(n, tensor.dtype); var u = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape); var p = eye(n, tensor.dtype); var swaps = 0
  if tensor.dtype == f32:
    let plData = cast[ptr UncheckedArray[float32]](addr BinaryBackend(l.data).buffer[0]); let puData = cast[ptr UncheckedArray[float32]](addr BinaryBackend(u.data).buffer[0]); let ppData = cast[ptr UncheckedArray[float32]](addr BinaryBackend(p.data).buffer[0])
    luImpl(float32)
  elif tensor.dtype == f64:
    let plData = cast[ptr UncheckedArray[float64]](addr BinaryBackend(l.data).buffer[0]); let puData = cast[ptr UncheckedArray[float64]](addr BinaryBackend(u.data).buffer[0]); let ppData = cast[ptr UncheckedArray[float64]](addr BinaryBackend(p.data).buffer[0])
    luImpl(float64)
  (p, l, u, swaps)

proc lu*(tensor: Tensor): (Tensor, Tensor, Tensor) = (let (p, l, u, _) = lu_internal(tensor); (p, l, u))

proc determinant*(tensor: Tensor): Tensor =
  let (p, l, u, swaps) = lu_internal(tensor); let n = tensor.shape[0]; var det: float64 = 1.0
  ops.withDataPtr(u, puData):
    for i in 0..<n: det *= puData[i * n + i].float64
  if swaps mod 2 != 0: det = -det
  let res = if tensor.dtype == f32: creation.tensor([det.float32], f32) else: creation.tensor([det], f64)
  return ops.reshape(res, @[])

proc cholesky*(tensor: Tensor): Tensor =
  let n = tensor.shape[0]; var l = zeros(@[n, n], tensor.dtype)
  if tensor.dtype == f32:
    let pSrc = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0]); let plData = cast[ptr UncheckedArray[float32]](addr BinaryBackend(l.data).buffer[0])
    for i in 0..<n:
      for j in 0..i:
        var s: float32 = 0; for k in 0..<j: s += plData[i * n + k] * plData[j * n + k]
        if i == j: plData[i * n + j] = nmath.sqrt(pSrc[i * n + i] - s)
        else: plData[i * n + j] = (1.0f32 / plData[j * n + j]) * (pSrc[i * n + j] - s)
  elif tensor.dtype == f64:
    let pSrc = cast[ptr UncheckedArray[float64]](addr BinaryBackend(tensor.data).buffer[0]); let plData = cast[ptr UncheckedArray[float64]](addr BinaryBackend(l.data).buffer[0])
    for i in 0..<n:
      for j in 0..i:
        var s: float64 = 0; for k in 0..<j: s += plData[i * n + k] * plData[j * n + k]
        if i == j: plData[i * n + j] = nmath.sqrt(pSrc[i * n + i] - s)
        else: plData[i * n + j] = (1.0 / plData[j * n + j]) * (pSrc[i * n + j] - s)
  l

proc qr*(tensor: Tensor): (Tensor, Tensor) =
  let n = tensor.shape[0]; let m = tensor.shape[1]; var q = zeros(@[n, m], tensor.dtype); var r = zeros(@[m, m], tensor.dtype)
  if tensor.dtype == f32:
    let pIn = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0]); let pQ = cast[ptr UncheckedArray[float32]](addr BinaryBackend(q.data).buffer[0]); let pR = cast[ptr UncheckedArray[float32]](addr BinaryBackend(r.data).buffer[0])
    for j in 0..<m:
      var v = newSeq[float32](n); for i in 0..<n: v[i] = pIn[i * m + j]
      for k in 0..<j:
        var s: float32 = 0; for i in 0..<n: s += pQ[i * m + k] * pIn[i * m + j]
        pR[k * m + j] = s; for i in 0..<n: v[i] -= s * pQ[i * m + k]
      var norm: float32 = 0; for i in 0..<n: norm += v[i] * v[i]
      norm = nmath.sqrt(norm); pR[j * m + j] = norm
      if norm > 1e-12: (for i in 0..<n: pQ[i * m + j] = v[i] / norm)
  elif tensor.dtype == f64:
    let pIn = cast[ptr UncheckedArray[float64]](addr BinaryBackend(tensor.data).buffer[0]); let pQ = cast[ptr UncheckedArray[float64]](addr BinaryBackend(q.data).buffer[0]); let pR = cast[ptr UncheckedArray[float64]](addr BinaryBackend(r.data).buffer[0])
    for j in 0..<m:
      var v = newSeq[float64](n); for i in 0..<n: v[i] = pIn[i * m + j]
      for k in 0..<j:
        var s: float64 = 0; for i in 0..<n: s += pQ[i * m + k] * pIn[i * m + j]
        pR[k * m + j] = s; for i in 0..<n: v[i] -= s * pQ[i * m + k]
      var norm: float64 = 0; for i in 0..<n: norm += v[i] * v[i]
      norm = nmath.sqrt(norm); pR[j * m + j] = norm
      if norm > 1e-12: (for i in 0..<n: pQ[i * m + j] = v[i] / norm)
  (q, r)

proc eigh*(tensor: Tensor): (Tensor, Tensor) =
  let n = tensor.shape[0]; var a = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape); var v = eye(n, tensor.dtype)
  if tensor.dtype == f32:
    let pA = cast[ptr UncheckedArray[float32]](addr BinaryBackend(a.data).buffer[0]); let pV = cast[ptr UncheckedArray[float32]](addr BinaryBackend(v.data).buffer[0])
    for iter in 0..<100:
      var maxVal: float32 = 0; var p, q = 0
      for i in 0..<n: (for j in i+1..<n: (if system.abs(pA[i * n + j]) > maxVal: (maxVal = system.abs(pA[i * n + j]); p = i; q = j)))
      if maxVal < 1e-9: break
      let theta = 0.5f32 * (pA[q * n + q] - pA[p * n + p]) / pA[p * n + q]
      let t = if theta >= 0: 1.0f32 / (theta + nmath.sqrt(1.0f32 + theta * theta)) else: -1.0f32 / (-theta + nmath.sqrt(1.0f32 + theta * theta))
      let c = 1.0f32 / nmath.sqrt(1.0f32 + t * t); let s = t * c
      let app = pA[p * n + p]; let aqq = pA[q * n + q]; let apq = pA[p * n + q]
      pA[p * n + p] = c * c * app - 2.0f32 * s * c * apq + s * s * aqq; pA[q * n + q] = s * s * app + 2.0f32 * s * c * apq + c * c * aqq; pA[p * n + q] = 0; pA[q * n + p] = 0
      for i in 0..<n: (if i != p and i != q: (let aip = pA[i * n + p]; let aiq = pA[i * n + q]; pA[i * n + p] = c * aip - s * aiq; pA[p * n + i] = pA[i * n + p]; pA[i * n + q] = s * aip + c * aiq; pA[q * n + i] = pA[i * n + q]))
      for i in 0..<n: (let vip = pV[i * n + p]; let viq = pV[i * n + q]; pV[i * n + p] = c * vip - s * viq; pV[i * n + q] = s * vip + c * viq)
  elif tensor.dtype == f64:
    let pA = cast[ptr UncheckedArray[float64]](addr BinaryBackend(a.data).buffer[0]); let pV = cast[ptr UncheckedArray[float64]](addr BinaryBackend(v.data).buffer[0])
    for iter in 0..<100:
      var maxVal: float64 = 0; var p, q = 0
      for i in 0..<n: (for j in i+1..<n: (if system.abs(pA[i * n + j]) > maxVal: (maxVal = system.abs(pA[i * n + j]); p = i; q = j)))
      if maxVal < 1e-9: break
      let theta = 0.5 * (pA[q * n + q] - pA[p * n + p]) / pA[p * n + q]
      let t = if theta >= 0: 1.0 / (theta + nmath.sqrt(1.0 + theta * theta)) else: -1.0 / (-theta + nmath.sqrt(1.0 + theta * theta))
      let c = 1.0 / nmath.sqrt(1.0 + t * t); let s = t * c
      let app = pA[p * n + p]; let aqq = pA[q * n + q]; let apq = pA[p * n + q]
      pA[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq; pA[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq; pA[p * n + q] = 0; pA[q * n + p] = 0
      for i in 0..<n: (if i != p and i != q: (let aip = pA[i * n + p]; let aiq = pA[i * n + q]; pA[i * n + p] = c * aip - s * aiq; pA[p * n + i] = pA[i * n + p]; pA[i * n + q] = s * aip + c * aiq; pA[q * n + i] = pA[i * n + q]))
      for i in 0..<n: (let vip = pV[i * n + p]; let viq = pV[i * n + q]; pV[i * n + p] = c * vip - s * viq; pV[i * n + q] = s * vip + c * viq)
  (take_diagonal(a), v)

proc svd*(tensor: Tensor): (Tensor, Tensor, Tensor) =
  let m = tensor.shape[0]; let n = tensor.shape[1]
  if m >= n:
    let at_a = ops.dot(ops.transpose(tensor), tensor)
    let (s_sq, v) = eigh(at_a)
    let s = ops.sqrt(s_sq)
    let s_inv = ops.divide(creation.ones(s.shape, s.dtype), s)
    let u = ops.dot(ops.dot(tensor, v), creation.diag(s_inv))
    return (u, s, ops.transpose(v))
  else:
    let (v, s, ut) = svd(ops.transpose(tensor))
    return (ops.transpose(ut), s, ops.transpose(v))

proc triangular_solve*(a, b: Tensor, lower: bool = true): Tensor =
  let n = a.shape[0]; let m = if b.shape.len > 1: b.shape[1] else: 1
  let res = creation.zeros(b.shape, b.dtype)
  ops.withDataPtr(a, pA):
    ops.withDataPtr(b, pB):
      ops.withDataPtr(res, pR):
        if lower:
          for i in 0..<n:
            for j in 0..<m:
              var s: float64 = 0
              for k in 0..<i: s += pA[i * n + k].float64 * pR[k * m + j].float64
              pR[i * m + j] = typeof(pR[0])((pB[i * m + j].float64 - s) / pA[i * n + i].float64)
        else:
          for i in countdown(n-1, 0):
            for j in 0..<m:
              var s: float64 = 0
              for k in i+1..<n: s += pA[i * n + k].float64 * pR[k * m + j].float64
              pR[i * m + j] = typeof(pR[0])((pB[i * m + j].float64 - s) / pA[i * n + i].float64)
  res

proc solve*(a, b: Tensor): Tensor =
  let (p, l, u, _) = lu_internal(a)
  let pb = ops.dot(ops.transpose(p), b)
  let y = triangular_solve(l, pb, lower = true)
  let x = triangular_solve(u, y, lower = false)
  x

proc matrix_power*(tensor: Tensor, n: int): Tensor =
  if tensor.shape.len != 2 or tensor.shape[0] != tensor.shape[1]: raise newException(ValueError, "matrix_power expects square matrix")
  if n == 0: return creation.eye(tensor.shape[0], tensor.dtype)
  if n == 1: return tensor
  if n < 0: return matrix_power(solve(tensor, creation.eye(tensor.shape[0], tensor.dtype)), -n)
  var res = creation.eye(tensor.shape[0], tensor.dtype)
  var base = tensor
  var p = n
  while p > 0:
    if p mod 2 == 1: res = ops.dot(res, base)
    base = ops.dot(base, base)
    p = p div 2
  res
