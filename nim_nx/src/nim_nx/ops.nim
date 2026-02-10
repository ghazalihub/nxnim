import types, backend, creation, shape
import std/math as nmath
import std/algorithm

proc broadcast*(left, right: Tensor): (Tensor, Tensor) =
  let newShape = broadcast_shape(left.shape, right.shape)
  (broadcast_to(left, newShape), broadcast_to(right, newShape))

proc add*(left, right: Tensor): Tensor =
  let (l_b, r_b) = broadcast(left, right)
  let t = zeros(l_b.shape, l_b.dtype)
  let totalSize = l_b.shape.size()
  let pl = cast[ptr UncheckedArray[float32]](addr BinaryBackend(l_b.data).buffer[0])
  let pr = cast[ptr UncheckedArray[float32]](addr BinaryBackend(r_b.data).buffer[0])
  let pres = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t.data).buffer[0])
  if l_b.dtype == f32:
    for i in 0..<totalSize: pres[i] = pl[i] + pr[i]
  elif l_b.dtype == s32:
    let pli = cast[ptr UncheckedArray[int32]](pl)
    let pri = cast[ptr UncheckedArray[int32]](pr)
    let presi = cast[ptr UncheckedArray[int32]](pres)
    for i in 0..<totalSize: presi[i] = pli[i] + pri[i]
  if left.requires_grad or right.requires_grad:
    t.requires_grad = true
    t.creator = Op(name: "add", inputs: @[left, right])
    t.creator.backward = proc(grad_output: Tensor) =
      if left.requires_grad:
        if left.grad == nil: left.grad = zeros(left.shape, left.dtype)
        left.grad = add(left.grad, grad_output)
      if right.requires_grad:
        if right.grad == nil: right.grad = zeros(right.shape, right.dtype)
        right.grad = add(right.grad, grad_output)
  t

proc multiply*(left, right: Tensor): Tensor =
  let (l_b, r_b) = broadcast(left, right)
  let t = zeros(l_b.shape, l_b.dtype)
  let totalSize = l_b.shape.size()
  let pl = cast[ptr UncheckedArray[float32]](addr BinaryBackend(l_b.data).buffer[0])
  let pr = cast[ptr UncheckedArray[float32]](addr BinaryBackend(r_b.data).buffer[0])
  let pres = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t.data).buffer[0])
  if l_b.dtype == f32:
    for i in 0..<totalSize: pres[i] = pl[i] * pr[i]
  if left.requires_grad or right.requires_grad:
    t.requires_grad = true
    t.creator = Op(name: "mul", inputs: @[left, right])
    t.creator.backward = proc(grad_output: Tensor) =
      if left.requires_grad:
        if left.grad == nil: left.grad = zeros(left.shape, left.dtype)
        left.grad = add(left.grad, multiply(grad_output, right))
      if right.requires_grad:
        if right.grad == nil: right.grad = zeros(right.shape, right.dtype)
        right.grad = add(right.grad, multiply(grad_output, left))
  t

proc backward*(tensor: Tensor) =
  if tensor.grad == nil: tensor.grad = ones(tensor.shape, tensor.dtype)
  if tensor.creator != nil:
    tensor.creator.backward(tensor.grad)
    for input in tensor.creator.inputs: backward(input)

proc subtract*(left, right: Tensor): Tensor =
  let (l_b, r_b) = broadcast(left, right)
  let t = zeros(l_b.shape, l_b.dtype)
  let totalSize = l_b.shape.size()
  let pl = cast[ptr UncheckedArray[float32]](addr BinaryBackend(l_b.data).buffer[0])
  let pr = cast[ptr UncheckedArray[float32]](addr BinaryBackend(r_b.data).buffer[0])
  let pres = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t.data).buffer[0])
  if l_b.dtype == f32:
    for i in 0..<totalSize: pres[i] = pl[i] - pr[i]
  t

proc divide*(left, right: Tensor): Tensor =
  let (l_b, r_b) = broadcast(left, right)
  let t = zeros(l_b.shape, l_b.dtype)
  let totalSize = l_b.shape.size()
  let pl = cast[ptr UncheckedArray[float32]](addr BinaryBackend(l_b.data).buffer[0])
  let pr = cast[ptr UncheckedArray[float32]](addr BinaryBackend(r_b.data).buffer[0])
  let pres = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t.data).buffer[0])
  if l_b.dtype == f32:
    for i in 0..<totalSize: pres[i] = pl[i] / pr[i]
  t

proc exp*(tensor: Tensor): Tensor =
  let t = zeros(tensor.shape, tensor.dtype)
  let totalSize = tensor.shape.size()
  let ps = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0])
  let pd = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t.data).buffer[0])
  if tensor.dtype == f32:
    for i in 0..<totalSize: pd[i] = nmath.exp(ps[i])
  if tensor.requires_grad:
    t.requires_grad = true
    t.creator = Op(name: "exp", inputs: @[tensor])
    t.creator.backward = proc(grad_output: Tensor) =
      if tensor.grad == nil: tensor.grad = zeros(tensor.shape, tensor.dtype)
      tensor.grad = add(tensor.grad, multiply(grad_output, t))
  t

proc sum*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor =
  let totalSize = tensor.shape.size()
  let ps = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0])
  var resVal: float32 = 0
  if tensor.dtype == f32:
    for i in 0..<totalSize: resVal += ps[i]
  let t = creation.tensor([resVal], tensor.dtype).reshape(if keep_dims: (var s = tensor.shape; for j in 0..<s.len: s[j]=1; s) else: @[])
  if tensor.requires_grad:
    t.requires_grad = true
    t.creator = Op(name: "sum", inputs: @[tensor])
    t.creator.backward = proc(grad_output: Tensor) =
      if tensor.grad == nil: tensor.grad = zeros(tensor.shape, tensor.dtype)
      tensor.grad = add(tensor.grad, broadcast_to(grad_output, tensor.shape))
  t

proc take*(tensor, indices: Tensor, axis: int = 0): Tensor =
  let iSize = indices.shape.size()
  let itemSize = tensor.dtype.bits div 8
  var resShape = tensor.shape
  resShape[axis] = iSize
  let res = zeros(resShape, tensor.dtype)
  let pIdx = cast[ptr UncheckedArray[int32]](addr BinaryBackend(indices.data).buffer[0])
  let pSrc = addr BinaryBackend(tensor.data).buffer[0]
  let pDst = addr BinaryBackend(res.data).buffer[0]
  let outerSize = if axis == 0: 1 else: tensor.shape[0..<axis].size()
  let axisSize = tensor.shape[axis]
  let innerSize = if axis == tensor.shape.len - 1: 1 else: tensor.shape[axis+1..^1].size()
  for o in 0..<outerSize:
    for i in 0..<iSize:
      let srcAxisIdx = pIdx[i]
      let srcBase = (o * axisSize + srcAxisIdx) * innerSize
      let dstBase = (o * iSize + i) * innerSize
      copyMem(cast[ptr byte](cast[int](pDst) + dstBase * itemSize),
              cast[ptr byte](cast[int](pSrc) + srcBase * itemSize),
              innerSize * itemSize)
  res

proc sort*(tensor: Tensor, axis: int = -1): Tensor =
  let res = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  if (axis == -1 or axis == 0) and tensor.shape.len == 1:
    let totalSize = tensor.shape[0]
    if tensor.dtype == f32:
      let p = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
      var s = newSeq[float32](totalSize)
      for i in 0..<totalSize: s[i] = p[i]
      s.sort()
      for i in 0..<totalSize: p[i] = s[i]
  res

proc argsort*(tensor: Tensor, axis: int = -1): Tensor =
  let totalSize = tensor.shape.size()
  var res = zeros(tensor.shape, s32)
  let pRes = cast[ptr UncheckedArray[int32]](addr BinaryBackend(res.data).buffer[0])
  if (axis == -1 or axis == 0) and tensor.shape.len == 1:
    if tensor.dtype == f32:
      let pSrc = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0])
      var s = newSeq[int32](totalSize)
      for i in 0..<totalSize: s[i] = i.int32
      s.sort(proc (x, y: int32): int = cmp(pSrc[x], pSrc[y]))
      for i in 0..<totalSize: pRes[i] = s[i]
  res

proc top_k*(tensor: Tensor, k: int, axis: int = -1): (Tensor, Tensor) =
  let totalSize = tensor.shape.size()
  let idx = argsort(tensor, axis)
  let pIdx = cast[ptr UncheckedArray[int32]](addr BinaryBackend(idx.data).buffer[0])
  var resVal = zeros(@[k], tensor.dtype)
  var resIdx = zeros(@[k], s32)
  let pResVal = cast[ptr UncheckedArray[float32]](addr BinaryBackend(resVal.data).buffer[0])
  let pResIdx = cast[ptr UncheckedArray[int32]](addr BinaryBackend(resIdx.data).buffer[0])
  let pSrc = cast[ptr UncheckedArray[float32]](addr BinaryBackend(tensor.data).buffer[0])
  for i in 0..<k:
    let srcIdx = pIdx[totalSize - 1 - i]
    pResIdx[i] = srcIdx
    if tensor.dtype == f32: pResVal[i] = pSrc[srcIdx]
  (resVal, resIdx)

proc conv*(input, filter: Tensor): Tensor =
  if input.shape.len == 1 and filter.shape.len == 1:
    let n = input.shape[0]
    let k = filter.shape[0]
    let resLen = n - k + 1
    var res = zeros(@[resLen], input.dtype)
    let pIn = cast[ptr UncheckedArray[float32]](addr BinaryBackend(input.data).buffer[0])
    let pFil = cast[ptr UncheckedArray[float32]](addr BinaryBackend(filter.data).buffer[0])
    let pRes = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
    for i in 0..<resLen:
      var s: float32 = 0
      for j in 0..<k: s += pIn[i + j] * pFil[j]
      pRes[i] = s
    return res
  input
