import types, creation, backend, ops, shape
import std/random
import std/math

proc key*(seed: int): Tensor =
  creation.tensor([0.uint32, seed.uint32], u32)

proc split*(key: Tensor, parts: int = 2): Tensor =
  var resShape = @[parts, 2]
  var t = creation.zeros(resShape, u32)
  let dst = BinaryBackend(t.data)
  let pDst = cast[ptr UncheckedArray[uint32]](addr dst.buffer[0])
  let src = BinaryBackend(key.data)
  let pSrc = cast[ptr UncheckedArray[uint32]](addr src.buffer[0])
  for i in 0..<parts:
    pDst[i * 2] = pSrc[0] + i.uint32 + 1
    pDst[i * 2 + 1] = pSrc[1] + i.uint32 + 100
  t

proc uniform*(key: Tensor, shape: seq[int], min_val = 0.0, max_val = 1.0): (Tensor, Tensor) =
  let totalSize = shape.size()
  var t = creation.zeros(shape, f32)
  let dst = BinaryBackend(t.data)
  let pd = cast[ptr UncheckedArray[float32]](addr dst.buffer[0])
  let kData = cast[ptr UncheckedArray[uint32]](addr BinaryBackend(key.data).buffer[0])
  var rng = initRand(kData[1].int64)
  for i in 0..<totalSize:
    pd[i] = rng.rand(min_val..max_val).float32
  let newKey = creation.tensor([kData[0], kData[1] + 1], u32)
  (t, newKey)

proc normal*(key: Tensor, shape: seq[int], mean = 0.0, std = 1.0): (Tensor, Tensor) =
  let (u1, k2) = uniform(key, shape)
  let (u2, k3) = uniform(k2, shape)
  let res = creation.zeros(shape, f32)
  let pU1 = cast[ptr UncheckedArray[float32]](addr BinaryBackend(u1.data).buffer[0])
  let pU2 = cast[ptr UncheckedArray[float32]](addr BinaryBackend(u2.data).buffer[0])
  let pRes = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
  for i in 0..<shape.size():
    let val1 = if pU1[i] == 0: 1e-10.float32 else: pU1[i]
    let r = sqrt(-2.0 * ln(val1))
    let theta = 2.0 * PI * pU2[i]
    pRes[i] = (r * cos(theta) * std + mean).float32
  (res, k3)

proc randint*(key: Tensor, min_val, max_val: int, shape: seq[int]): (Tensor, Tensor) =
  let totalSize = shape.size()
  var t = creation.zeros(shape, s32)
  let dst = BinaryBackend(t.data)
  let pd = cast[ptr UncheckedArray[int32]](addr dst.buffer[0])
  let kData = cast[ptr UncheckedArray[uint32]](addr BinaryBackend(key.data).buffer[0])
  var rng = initRand(kData[1].int64)
  for i in 0..<totalSize:
    pd[i] = rng.rand(min_val..max_val-1).int32
  let newKey = creation.tensor([kData[0], kData[1] + 1], u32)
  (t, newKey)

proc shuffle*(key: Tensor, tensor: Tensor, axis: int = 0): (Tensor, Tensor) =
  let res = creation.from_binary(BinaryBackend(tensor.data).buffer, tensor.dtype, tensor.shape)
  let dst = BinaryBackend(res.data)
  let n = tensor.shape[axis]
  let kData = cast[ptr UncheckedArray[uint32]](addr BinaryBackend(key.data).buffer[0])
  var rng = initRand(kData[1].int64)
  if tensor.shape.len == 1:
    let p = cast[ptr UncheckedArray[float32]](addr dst.buffer[0])
    for i in countdown(n-1, 1):
      let j = rng.rand(i)
      let tmp = p[i]
      p[i] = p[j]
      p[j] = tmp
  (res, creation.tensor([kData[0], kData[1] + 1], u32))

proc choice*(key: Tensor, tensor: Tensor, samples: int): (Tensor, Tensor) =
  let kData = cast[ptr UncheckedArray[uint32]](addr BinaryBackend(key.data).buffer[0])
  var rng = initRand(kData[1].int64)
  let n = tensor.shape[0]
  var idxs = newSeq[int32](samples)
  for i in 0..<samples:
    idxs[i] = rng.rand(n-1).int32
  let resIdx = creation.tensor(idxs, s32)
  (ops.take(tensor, resIdx, 0), creation.tensor([kData[0], kData[1] + 1], u32))

proc gumbel*(key: Tensor, shape: seq[int]): (Tensor, Tensor) =
  let (u, k2) = uniform(key, shape)
  let pU = cast[ptr UncheckedArray[float32]](addr BinaryBackend(u.data).buffer[0])
  let res = creation.zeros(shape, f32)
  let pRes = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
  for i in 0..<shape.size():
    let val = if pU[i] == 0: 1e-10.float32 else: pU[i]
    pRes[i] = -ln(-ln(val))
  (res, k2)
