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
      elif dtype == s32:
        var val = data[i].int32
        copyMem(addr bytes[i*4], addr val, 4)
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
  let totalSize = shape.size()
  Tensor(data: newBinaryBackend(totalSize * itemSize), shape: shape, dtype: dtype)

proc iota*(shape: seq[int], axis: Option[int] = none(int), dtype: DType = s32): Tensor =
  let totalSize = shape.size()
  let itemSize = dtype.bits div 8
  var bytes = newSeq[byte](totalSize * itemSize)
  for i in 0..<totalSize:
    if dtype == s32:
      var val = i.int32
      copyMem(addr bytes[i*4], addr val, 4)
  Tensor(data: newBinaryBackend(bytes), shape: shape, dtype: dtype)

proc eye*(n: int, dtype: DType = f32): Tensor =
  let shape = @[n, n]
  var t = zeros(shape, dtype)
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  for i in 0..<n:
    if dtype == f32:
      var val = 1.0.float32
      copyMem(addr backend.buffer[(i * n + i) * itemSize], addr val, itemSize)
  t

proc linspace*(start, stop: float64, n: int, dtype: DType = f32): Tensor =
  let shape = @[n]
  var t = zeros(shape, dtype)
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  let step = (stop - start) / (n - 1).float64
  for i in 0..<n:
    if dtype == f32:
      var val = (start + i.float64 * step).float32
      copyMem(addr backend.buffer[i * itemSize], addr val, itemSize)
  t

proc full*(shape: seq[int], value: float64, dtype: DType = f32): Tensor =
  var t = zeros(shape, dtype)
  let totalSize = shape.size()
  let itemSize = dtype.bits div 8
  let backend = BinaryBackend(t.data)
  for i in 0..<totalSize:
    if dtype == f32:
      var val = value.float32
      copyMem(addr backend.buffer[i * itemSize], addr val, itemSize)
    elif dtype == s32:
      var val = value.int32
      copyMem(addr backend.buffer[i * itemSize], addr val, itemSize)
  t

proc ones*(shape: seq[int], dtype: DType = f32): Tensor =
  full(shape, 1.0, dtype)
