import types, backend, creation
import std/sequtils

proc reshape*(tensor: Tensor, newShape: seq[int]): Tensor =
  if tensor.shape.size() != newShape.size():
    raise newException(ValueError, "Cannot reshape: total size must remain constant")
  Tensor(data: tensor.data, shape: newShape, dtype: tensor.dtype, names: tensor.names)

proc flatten*(tensor: Tensor): Tensor =
  reshape(tensor, @[tensor.shape.size()])

proc transpose*(tensor: Tensor, axes: seq[int] = @[]): Tensor =
  if tensor.shape.len != 2: return tensor
  let n = tensor.shape[0]
  let m = tensor.shape[1]
  var t = zeros(@[m, n], tensor.dtype)
  let itemSize = tensor.dtype.bits div 8
  let src = BinaryBackend(tensor.data)
  let dst = BinaryBackend(t.data)
  for i in 0..<n:
    for j in 0..<m:
      copyMem(addr dst.buffer[(j * n + i) * itemSize], addr src.buffer[(i * m + j) * itemSize], itemSize)
  t

proc broadcast_shape*(s1, s2: seq[int]): seq[int] =
  let l1 = s1.len
  let l2 = s2.len
  let maxLen = max(l1, l2)
  result = newSeq[int](maxLen)
  for i in 1..maxLen:
    let d1 = if i <= l1: s1[l1 - i] else: 1
    let d2 = if i <= l2: s2[l2 - i] else: 1
    if d1 == d2:
      result[maxLen - i] = d1
    elif d1 == 1:
      result[maxLen - i] = d2
    elif d2 == 1:
      result[maxLen - i] = d1
    else:
      raise newException(ValueError, "Incompatible shapes for broadcasting")

proc broadcast_to*(tensor: Tensor, shape: seq[int]): Tensor =
  if tensor.shape == shape: return tensor
  let tShape = tensor.shape
  let tLen = tShape.len
  let sLen = shape.len
  if tLen > sLen: raise newException(ValueError, "Cannot broadcast to fewer dimensions")
  for i in 1..tLen:
    let d1 = tShape[tLen - i]
    let d2 = shape[sLen - i]
    if d1 != d2 and d1 != 1:
      raise newException(ValueError, "Incompatible dimensions for broadcast_to")

  let res = zeros(shape, tensor.dtype)
  let totalSize = shape.size()
  let itemSize = tensor.dtype.bits div 8
  let src = BinaryBackend(tensor.data)
  let dst = BinaryBackend(res.data)

  for i in 0..<totalSize:
    var subIdx = newSeq[int](sLen)
    var rem = i
    for j in countdown(sLen - 1, 0):
      subIdx[j] = rem mod shape[j]
      rem = rem div shape[j]

    var srcIdx = 0
    var stride = 1
    for j in countdown(tLen - 1, 0):
      let d1 = tShape[j]
      let si = subIdx[sLen - (tLen - j)]
      if d1 != 1:
        srcIdx += si * stride
      stride *= d1

    copyMem(addr dst.buffer[i * itemSize], addr src.buffer[srcIdx * itemSize], itemSize)
  res

proc new_axis*(tensor: Tensor, axis: int): Tensor =
  var newShape = tensor.shape
  newShape.insert(1, axis)
  reshape(tensor, newShape)

proc squeeze*(tensor: Tensor, axes: seq[int] = @[]): Tensor =
  var newShape: seq[int] = @[]
  for i, s in tensor.shape:
    if s != 1 or (axes.len > 0 and i notin axes):
      newShape.add(s)
  reshape(tensor, newShape)

proc concatenate*(tensors: seq[Tensor], axis: int = 0): Tensor =
  if tensors.len == 0: raise newException(ValueError, "No tensors to concatenate")
  let firstShape = tensors[0].shape
  var resShape = firstShape
  var totalAxisDim = 0
  for t in tensors:
    if t.shape.len != firstShape.len: raise newException(ValueError, "Ranks must match")
    for i in 0..<firstShape.len:
      if i != axis and t.shape[i] != firstShape[i]:
        raise newException(ValueError, "Dimensions must match except for concatenate axis")
    totalAxisDim += t.shape[axis]
  resShape[axis] = totalAxisDim

  let res = zeros(resShape, tensors[0].dtype)
  let itemSize = res.dtype.bits div 8
  let dst = BinaryBackend(res.data)

  var offset = 0
  for t in tensors:
    let tSize = t.shape.size()
    let src = BinaryBackend(t.data)
    for i in 0..<tSize:
      var subIdx = newSeq[int](t.shape.len)
      var rem = i
      for j in countdown(t.shape.len - 1, 0):
        subIdx[j] = rem mod t.shape[j]
        rem = rem div t.shape[j]

      subIdx[axis] += offset

      var dstIdx = 0
      var stride = 1
      for j in countdown(resShape.len - 1, 0):
        dstIdx += subIdx[j] * stride
        stride *= resShape[j]

      copyMem(addr dst.buffer[dstIdx * itemSize], addr src.buffer[i * itemSize], itemSize)
    offset += t.shape[axis]
  res

proc stack*(tensors: seq[Tensor], axis: int = 0): Tensor =
  var reshaped: seq[Tensor] = @[]
  for t in tensors:
    reshaped.add(t.new_axis(axis))
  concatenate(reshaped, axis)

proc slice*(tensor: Tensor, starts: seq[int], lengths: seq[int]): Tensor =
  let res = zeros(lengths, tensor.dtype)
  let itemSize = tensor.dtype.bits div 8
  let src = BinaryBackend(tensor.data)
  let dst = BinaryBackend(res.data)
  let totalSize = lengths.size()

  for i in 0..<totalSize:
    var subIdx = newSeq[int](lengths.len)
    var rem = i
    for j in countdown(lengths.len - 1, 0):
      subIdx[j] = rem mod lengths[j]
      rem = rem div lengths[j]

    var srcIdx = 0
    var stride = 1
    for j in countdown(tensor.shape.len - 1, 0):
      srcIdx += (subIdx[j] + starts[j]) * stride
      stride *= tensor.shape[j]

    copyMem(addr dst.buffer[i * itemSize], addr src.buffer[srcIdx * itemSize], itemSize)
  res

proc put_slice*(tensor: Tensor, starts: seq[int], slice: Tensor): Tensor =
  let res = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  let dst = BinaryBackend(res.data)
  let src = BinaryBackend(slice.data)
  let sliceShape = slice.shape
  let totalSize = sliceShape.size()
  let itemSize = tensor.dtype.bits div 8

  for i in 0..<totalSize:
    var subIdx = newSeq[int](sliceShape.len)
    var rem = i
    for j in countdown(sliceShape.len - 1, 0):
      subIdx[j] = rem mod sliceShape[j]
      rem = rem div sliceShape[j]

    var dstIdx = 0
    var stride = 1
    for j in countdown(tensor.shape.len - 1, 0):
      dstIdx += (subIdx[j] + starts[j]) * stride
      stride *= tensor.shape[j]

    copyMem(addr dst.buffer[dstIdx * itemSize], addr src.buffer[i * itemSize], itemSize)
  res

proc tile*(tensor: Tensor, reps: seq[int]): Tensor =
  var resShape = tensor.shape
  for i in 0..<resShape.len: resShape[i] *= reps[i]
  let res = zeros(resShape, tensor.dtype)
  let totalSize = resShape.size()
  let itemSize = tensor.dtype.bits div 8
  let src = BinaryBackend(tensor.data)
  let dst = BinaryBackend(res.data)
  for i in 0..<totalSize:
    var subIdx = newSeq[int](resShape.len)
    var rem = i
    for j in countdown(resShape.len - 1, 0):
      subIdx[j] = rem mod resShape[j]
      rem = rem div resShape[j]
    var srcSubIdx = newSeq[int](tensor.shape.len)
    for j in 0..<tensor.shape.len:
      srcSubIdx[j] = subIdx[j] mod tensor.shape[j]
    var srcIdx = 0
    var stride = 1
    for j in countdown(tensor.shape.len - 1, 0):
      srcIdx += srcSubIdx[j] * stride
      stride *= tensor.shape[j]
    copyMem(addr dst.buffer[i * itemSize], addr src.buffer[srcIdx * itemSize], itemSize)
  res
