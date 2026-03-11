import types, creation, backend, ops
import std/random
import std/hashes

type
  RandomKey* = Tensor # Seed stored as s64 to match Nx behavior

proc key*(seed: int): RandomKey =
  creation.tensor(@[seed.int64], s64)

proc split*(key: RandomKey, num: int = 2): seq[RandomKey] =
  let seed = cast[ptr int64](addr BinaryBackend(key.data).buffer[0])[]
  result = newSeq[RandomKey](num)
  for i in 0..<num:
    # Use hashing to create more independent seeds
    var h = hash(seed)
    h = h !& hash(i)
    result[i] = creation.tensor(@[h.int64], s64)

proc fold_in*(key: RandomKey, data: int): RandomKey =
  let seed = cast[ptr int64](addr BinaryBackend(key.data).buffer[0])[]
  var h = hash(seed)
  h = h !& hash(data)
  creation.tensor(@[h.int64], s64)

proc uniform*(key: RandomKey, shape: seq[int], minVal = 0.0f64, maxVal = 1.0f64, dtype: DType = f32): (Tensor, RandomKey) =
  let seed = cast[ptr int64](addr BinaryBackend(key.data).buffer[0])[]
  var rng = initRand(seed)
  let sz = types.size(shape)
  let res = creation.zeros(shape, dtype)
  let bO = BinaryBackend(res.data)
  if dtype == f32:
    let p = cast[ptr UncheckedArray[float32]](addr bO.buffer[0])
    for i in 0..<sz: p[i] = rng.rand(maxVal.float32 - minVal.float32) + minVal.float32
  elif dtype == f64:
    let p = cast[ptr UncheckedArray[float64]](addr bO.buffer[0])
    for i in 0..<sz: p[i] = rng.rand(maxVal - minVal) + minVal

  let nextKey = fold_in(key, 1)
  (res, nextKey)

proc normal*(key: RandomKey, shape: seq[int], mu = 0.0f64, sigma = 1.0f64, dtype: DType = f32): (Tensor, RandomKey) =
  let seed = cast[ptr int64](addr BinaryBackend(key.data).buffer[0])[]
  var rng = initRand(seed)
  let sz = types.size(shape)
  let res = creation.zeros(shape, dtype)
  let bO = BinaryBackend(res.data)
  if dtype == f32:
    let p = cast[ptr UncheckedArray[float32]](addr bO.buffer[0])
    for i in 0..<sz: p[i] = rng.gauss(mu, sigma).float32
  elif dtype == f64:
    let p = cast[ptr UncheckedArray[float64]](addr bO.buffer[0])
    for i in 0..<sz: p[i] = rng.gauss(mu, sigma)

  let nextKey = fold_in(key, 1)
  (res, nextKey)
