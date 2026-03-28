type
  DTypeKind* = enum
    dtS, dtU, dtF, dtBF, dtC, dtF8E4M3FN

  DType* = object
    kind*: DTypeKind
    bits*: int

  BackendData* = ref object of RootObj

  Tensor* = ref object
    data*: BackendData
    shape*: seq[int]
    dtype*: DType
    names*: seq[string]
    # Autograd fields
    grad*: Tensor
    requires_grad*: bool
    creator*: Op

  Op* = ref object
    name*: string
    inputs*: seq[Tensor]
    backward*: proc(grad_output: Tensor)

proc `$`*(dtype: DType): string =
  case dtype.kind:
  of dtS: "s" & $dtype.bits
  of dtU: "u" & $dtype.bits
  of dtF: "f" & $dtype.bits
  of dtBF: "bf" & $dtype.bits
  of dtC: "c" & $dtype.bits
  of dtF8E4M3FN: "f8_e4m3fn"

proc size*(shape: seq[int]): int =
  if shape.len == 0: return 1
  result = 1
  for s in shape: result *= s

const
  s8* = DType(kind: dtS, bits: 8)
  s16* = DType(kind: dtS, bits: 16)
  s32* = DType(kind: dtS, bits: 32)
  s64* = DType(kind: dtS, bits: 64)
  u8* = DType(kind: dtU, bits: 8)
  u16* = DType(kind: dtU, bits: 16)
  u32* = DType(kind: dtU, bits: 32)
  u64* = DType(kind: dtU, bits: 64)
  f16* = DType(kind: dtF, bits: 16)
  f32* = DType(kind: dtF, bits: 32)
  f64* = DType(kind: dtF, bits: 64)
  bf16* = DType(kind: dtBF, bits: 16)
  c64* = DType(kind: dtC, bits: 64)
  c128* = DType(kind: dtC, bits: 128)
