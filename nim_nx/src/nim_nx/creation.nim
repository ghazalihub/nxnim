import types, backend
import std/sequtils
import std/options

proc tensor*[T](data: openArray[T], dtype: DType = f32): Tensor =
  let shape = @[data.len]
  let itemSize = dtype.bits div 8
  var bytes = newSeq[byte](data.len * itemSize)
  if data.len > 0:
    for i in 0..<data.len:
      if dtype == f32:
        var val = data[i].float32
        copyMem(addr bytes[i*4], addr val, 4)
      elif dtype == f64:
        var val = data[i].float64
        copyMem(addr bytes[i*8], addr val, 8)
      elif dtype == s32:
        var val = data[i].int32
        copyMem(addr bytes[i*4], addr val, 4)
      elif dtype == s64:
        var val = data[i].int64
        copyMem(addr bytes[i*8], addr val, 8)
      elif dtype == u8:
        var val = data[i].uint8
        copyMem(addr bytes[i], addr val, 1)
      else:
        copyMem(addr bytes[i * itemSize], unsafeAddr data[i], itemSize)
  Tensor(data: newBinaryBackend(bytes), shape: shape, dtype: dtype)

proc from_binary*(binary: seq[byte], dtype: DType, shape: seq[int]): Tensor =
  Tensor(data: newBinaryBackend(binary), shape: shape, dtype: dtype)

proc zeros*(shape: seq[int], dtype: DType = f32): Tensor =
  let itemSize = dtype.bits div 8
  Tensor(data: newBinaryBackend(shape.size() * itemSize), shape: shape, dtype: dtype)

proc iota*(shape: seq[int], axis: Option[int] = none(int), dtype: DType = s32): Tensor =
  let totalSize = shape.size()
  let itemSize = dtype.bits div 8
  var bytes = newSeq[byte](totalSize * itemSize)
  if axis.isNone:
    for i in 0..<totalSize:
      if dtype == s32:
        var val = i.int32; copyMem(addr bytes[i*4], addr val, 4)
      elif dtype == s64:
        var val = i.int64; copyMem(addr bytes[i*8], addr val, 8)
      elif dtype == f32:
        var val = i.float32; copyMem(addr bytes[i*4], addr val, 4)
  else:
    let ax = axis.get()
    let innerSize = if ax == shape.len-1: 1 else: shape[ax+1..^1].size()
    let axisDim = shape[ax]
    let outerSize = totalSize div (axisDim * innerSize)
    for o in 0..<outerSize:
      for i in 0..<axisDim:
        for inIdx in 0..<innerSize:
          let idx = (o * axisDim + i) * innerSize + inIdx
          if dtype == s32:
            var val = i.int32; copyMem(addr bytes[idx*4], addr val, 4)
          elif dtype == f32:
            var val = i.float32; copyMem(addr bytes[idx*4], addr val, 4)
  Tensor(data: newBinaryBackend(bytes), shape: shape, dtype: dtype)

proc full*(shape: seq[int], value: float64, dtype: DType = f32): Tensor =
  var t = zeros(shape, dtype)
  let totalSize = shape.size()
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  for i in 0..<totalSize:
    if dtype == f32:
      var val = value.float32; copyMem(addr backend.buffer[i*itemSize], addr val, 4)
    elif dtype == f64:
      var val = value.float64; copyMem(addr backend.buffer[i*itemSize], addr val, 8)
    elif dtype == s32:
      var val = value.int32; copyMem(addr backend.buffer[i*itemSize], addr val, 4)
    elif dtype == s64:
      var val = value.int64; copyMem(addr backend.buffer[i*itemSize], addr val, 8)
    elif dtype == u8:
      var val = value.uint8; copyMem(addr backend.buffer[i*itemSize], addr val, 1)
  t

proc eye*(n: int, dtype: DType = f32): Tensor =
  let t = zeros(@[n, n], dtype)
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  for i in 0..<n:
    if dtype == f32:
      var val = 1.0f32; copyMem(addr backend.buffer[(i*n+i)*itemSize], addr val, 4)
    elif dtype == f64:
      var val = 1.0; copyMem(addr backend.buffer[(i*n+i)*itemSize], addr val, 8)
    elif dtype == s32:
      var val = 1i32; copyMem(addr backend.buffer[(i*n+i)*itemSize], addr val, 4)
  t

proc linspace*(start, stop: float64, n: int, dtype: DType = f32): Tensor =
  let t = zeros(@[n], dtype)
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  let step = if n > 1: (stop - start) / (n - 1).float64 else: 0.0
  for i in 0..<n:
    let v = start + i.float64 * step
    if dtype == f32:
      var val = v.float32; copyMem(addr backend.buffer[i*itemSize], addr val, 4)
    elif dtype == f64:
      var val = v.float64; copyMem(addr backend.buffer[i*itemSize], addr val, 8)
  t

proc fill*(tensor: Tensor, value: float64): Tensor = full(tensor.shape, value, tensor.dtype)
proc ones*(shape: seq[int], dtype: DType = f32): Tensor = full(shape, 1.0, dtype)

proc tri*(n, m: int, k: int = 0, dtype: DType = f32): Tensor =
  let t = zeros(@[n, m], dtype)
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  for i in 0..<n:
    for j in 0..<m:
      if j <= i + k:
        if dtype == f32:
          var val = 1.0f32; copyMem(addr backend.buffer[(i*m+j)*itemSize], addr val, 4)
        elif dtype == f64:
          var val = 1.0; copyMem(addr backend.buffer[(i*m+j)*itemSize], addr val, 8)
        elif dtype == s32:
          var val = 1i32; copyMem(addr backend.buffer[(i*m+j)*itemSize], addr val, 4)
  t

proc tril*(tensor: Tensor, k: int = 0): Tensor =
  if tensor.shape.len < 2: raise newException(ValueError, "tril expects at least 2 dimensions")
  let n = tensor.shape[^2]; let m = tensor.shape[^1]
  let res = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  let bkd = BinaryBackend(res.data)
  let itemSize = tensor.dtype.bits div 8
  let outerSize = tensor.shape.size() div (n * m)
  for o in 0..<outerSize:
    for i in 0..<n:
      for j in 0..<m:
        if j > i + k:
          var zero: byte = 0
          for b in 0..<itemSize: bkd.buffer[(o*n*m+i*m+j)*itemSize+b] = zero
  res

proc triu*(tensor: Tensor, k: int = 0): Tensor =
  if tensor.shape.len < 2: raise newException(ValueError, "triu expects at least 2 dimensions")
  let n = tensor.shape[^2]; let m = tensor.shape[^1]
  let res = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  let bkd = BinaryBackend(res.data)
  let itemSize = tensor.dtype.bits div 8
  let outerSize = tensor.shape.size() div (n * m)
  for o in 0..<outerSize:
    for i in 0..<n:
      for j in 0..<m:
        if j < i + k:
          var zero: byte = 0
          for b in 0..<itemSize: bkd.buffer[(o*n*m+i*m+j)*itemSize+b] = zero
  res

proc to_binary*(tensor: Tensor): seq[uint8] = BinaryBackend(tensor.data).buffer

proc diag*(tensor: Tensor): Tensor =
  if tensor.shape.len == 1:
    let n = tensor.shape[0]
    let res = zeros(@[n, n], tensor.dtype)
    let itemSize = tensor.dtype.bits div 8
    let bkdIn = BinaryBackend(tensor.data)
    let bkdOut = BinaryBackend(res.data)
    for i in 0..<n: copyMem(addr bkdOut.buffer[(i*n+i)*itemSize], addr bkdIn.buffer[i*itemSize], itemSize)
    return res
  elif tensor.shape.len == 2:
    let n = system.min(tensor.shape[0], tensor.shape[1])
    let res = zeros(@[n], tensor.dtype)
    let itemSize = tensor.dtype.bits div 8
    let m = tensor.shape[1]
    let bkdIn = BinaryBackend(tensor.data)
    let bkdOut = BinaryBackend(res.data)
    for i in 0..<n: copyMem(addr bkdOut.buffer[i*itemSize], addr bkdIn.buffer[(i*m+i)*itemSize], itemSize)
    return res
  else: raise newException(ValueError, "diag expects 1D or 2D tensor")

proc make_diagonal*(tensor: Tensor): Tensor = diag(tensor)

proc take_diagonal*(tensor: Tensor): Tensor = diag(tensor)

proc put_diagonal*(tensor, diagonal: Tensor): Tensor =
  let res = from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  let n = system.min(tensor.shape[0], tensor.shape[1])
  let itemSize = tensor.dtype.bits div 8
  let m = tensor.shape[1]
  let bkdDiag = BinaryBackend(diagonal.data)
  let bkdRes = BinaryBackend(res.data)
  for i in 0..<n: copyMem(addr bkdRes.buffer[(i*m+i)*itemSize], addr bkdDiag.buffer[i*itemSize], itemSize)
  res
