## Core operations for Nim Nx.
import types, backend, creation, shape as nshape
import std/math as nmath
import std/algorithm
import std/json
import std/sets
import std/sequtils
import std/bitops

var grad_enabled* = true

# Forward declarations
proc add*(left, right: Tensor): Tensor
proc multiply*(left, right: Tensor): Tensor
proc subtract*(left, right: Tensor): Tensor
proc divide*(left, right: Tensor): Tensor
proc negate*(tensor: Tensor): Tensor
proc sum*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor
proc sin*(tensor: Tensor): Tensor
proc cos*(tensor: Tensor): Tensor
proc exp*(tensor: Tensor): Tensor
proc log*(tensor: Tensor): Tensor
proc sigmoid*(tensor: Tensor): Tensor
proc reshape*(tensor: Tensor, newShape: seq[int]): Tensor
proc transpose*(tensor: Tensor, axes: seq[int] = @[]): Tensor
proc broadcast_to*(tensor: Tensor, newShape: seq[int]): Tensor
proc dot*(left, right: Tensor): Tensor
proc sort*(tensor: Tensor, axis: int = -1): Tensor
proc take*(tensor, indices: Tensor, axis: int = 0): Tensor
proc as_type*(tensor: Tensor, dtype: DType): Tensor
proc to_number*(tensor: Tensor): float32
proc sqrt*(tensor: Tensor): Tensor
proc all*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor
proc any*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor
proc indexed_add*(t, indices, updates: Tensor): Tensor
proc outer*(left, right: Tensor): Tensor
proc select*(pred, on_true, on_false: Tensor): Tensor

# --- Templates ---

template withDataPtr*(t: Tensor, pName: untyped, body: untyped) =
  let bkd = BinaryBackend(t.data)
  if bkd.buffer.len > 0:
    if t.dtype == types.f32:
      let pName {.inject.} = cast[ptr UncheckedArray[float32]](addr bkd.buffer[0]); body
    elif t.dtype == types.f64:
      let pName {.inject.} = cast[ptr UncheckedArray[float64]](addr bkd.buffer[0]); body
    elif t.dtype == types.s32:
      let pName {.inject.} = cast[ptr UncheckedArray[int32]](addr bkd.buffer[0]); body
    elif t.dtype == types.s64:
      let pName {.inject.} = cast[ptr UncheckedArray[int64]](addr bkd.buffer[0]); body
    elif t.dtype == types.u8:
      let pName {.inject.} = cast[ptr UncheckedArray[uint8]](addr bkd.buffer[0]); body

template applyUnary(tIn: Tensor, outDT: DType, op: untyped): Tensor =
  let sz = types.size(tIn.shape); let resT = creation.zeros(tIn.shape, outDT)
  if sz > 0:
    withDataPtr(tIn, pI):
      withDataPtr(resT, pO):
        for i in 0..<sz:
          let val {.inject.} = pI[i].float64
          pO[i] = typeof(pO[0])(op)
  resT

template applyBinary(tL, tR: Tensor, outDT: DType, op: untyped): Tensor =
  let bShp = nshape.broadcast_shape(tL.shape, tR.shape)
  let lb = nshape.broadcast_to(tL, bShp)
  let rb = nshape.broadcast_to(tR, bShp)
  let resT = creation.zeros(bShp, outDT); let sz = types.size(bShp)
  if sz > 0:
    withDataPtr(lb, pL):
      withDataPtr(rb, pR):
        withDataPtr(resT, pO):
          for i in 0..<sz:
            let a {.inject.} = pL[i].float64
            let b {.inject.} = pR[i].float64
            pO[i] = typeof(pO[0])(op)
  resT

# --- Autograd Utilities ---

proc maybe_add_grad(t: Tensor, g: Tensor) =
  if t != nil and t.requires_grad:
    let old_ge = grad_enabled; grad_enabled = false
    if t.grad == nil: t.grad = g
    else: t.grad = add(t.grad, g)
    grad_enabled = old_ge

# --- Utility ---

proc zeros_like*(tensor: Tensor): Tensor = creation.zeros(tensor.shape, tensor.dtype)
proc ones_like*(tensor: Tensor): Tensor = creation.ones(tensor.shape, tensor.dtype)

proc with_grad*(tensor: Tensor, requires_grad: bool = true): Tensor =
  tensor.requires_grad = requires_grad
  tensor

proc to_scalar_float(tensor: Tensor): float64 =
  withDataPtr(tensor, p): return p[0].float64
  return 0.0

proc to_number*(tensor: Tensor): float32 = tensor.to_scalar_float().float32

proc to_list*(tensor: Tensor): JsonNode =
  if tensor == nil: return newJNull()
  let shp = tensor.shape; let sz = types.size(shp)
  if sz == 0: return newJArray()
  proc to_nested_list(offset: int, dim: int): JsonNode =
    if dim == shp.len:
      withDataPtr(tensor, p): return %(p[offset].float64)
      return newJNull()
    var res = newJArray(); let stride = if dim + 1 < shp.len: types.size(shp[(dim+1)..^1]) else: 1
    for i in 0..<shp[dim]: res.add(to_nested_list(offset + i * stride, dim + 1))
    res
  to_nested_list(0, 0)

proc `$`*(tensor: Tensor): string =
  if tensor == nil: "nil"
  else: "Tensor(" & $tensor.shape & ", " & $tensor.dtype & ")\n" & $to_list(tensor)

proc as_type*(tensor: Tensor, dtype: DType): Tensor =
  if tensor.dtype == dtype: return tensor
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, dtype)
  if sz > 0:
    withDataPtr(tensor, pI):
      withDataPtr(res, pO):
        for i in 0..<sz: pO[i] = typeof(pO[0])(pI[i].float64)
  return res

# --- Ops ---

proc sin*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, types.f32, nmath.sin(val))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "sin", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, multiply(g, cos(tensor))))
  res

proc cos*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, types.f32, nmath.cos(val))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "cos", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, multiply(g, negate(sin(tensor)))))
  res

proc exp*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, types.f32, nmath.exp(val))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "exp", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, multiply(g, res)))
  res

proc log*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, types.f32, nmath.ln(val))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "log", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, divide(g, tensor)))
  res

proc abs*(tensor: Tensor): Tensor = applyUnary(tensor, tensor.dtype, system.abs(val))

proc negate*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, tensor.dtype, -val)
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "negate", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, negate(g)))
  res

proc sqrt*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, types.f32, nmath.sqrt(val))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "sqrt", inputs: @[tensor], backward: proc(g: Tensor) =
      let two = creation.full(tensor.shape, 2.0, tensor.dtype)
      maybe_add_grad(tensor, divide(g, multiply(two, res))))
  res

proc sigmoid*(tensor: Tensor): Tensor =
  let res = applyUnary(tensor, types.f32, 1.0/(1.0+nmath.exp(-val)))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "sigmoid", inputs: @[tensor], backward: proc(g: Tensor) =
      let one = creation.ones(res.shape, res.dtype)
      maybe_add_grad(tensor, multiply(g, multiply(res, subtract(one, res)))))
  res

proc logical_not*(tensor: Tensor): Tensor = applyUnary(tensor, types.u8, (if val == 0: 1.0 else: 0.0))

proc add*(left, right: Tensor): Tensor =
  let res = applyBinary(left, right, left.dtype, a + b)
  if grad_enabled and (left.requires_grad or right.requires_grad):
    res.requires_grad = true
    res.creator = Op(name: "add", inputs: @[left, right], backward: proc(g: Tensor) =
      maybe_add_grad(left, g)
      maybe_add_grad(right, g))
  res

proc subtract*(left, right: Tensor): Tensor =
  let res = applyBinary(left, right, left.dtype, a - b)
  if grad_enabled and (left.requires_grad or right.requires_grad):
    res.requires_grad = true
    res.creator = Op(name: "subtract", inputs: @[left, right], backward: proc(g: Tensor) =
      maybe_add_grad(left, g)
      maybe_add_grad(right, negate(g)))
  res

proc multiply*(left, right: Tensor): Tensor =
  let res = applyBinary(left, right, left.dtype, a * b)
  if grad_enabled and (left.requires_grad or right.requires_grad):
    res.requires_grad = true
    res.creator = Op(name: "multiply", inputs: @[left, right], backward: proc(g: Tensor) =
      maybe_add_grad(left, multiply(g, right))
      maybe_add_grad(right, multiply(g, left)))
  res

proc divide*(left, right: Tensor): Tensor =
  let outDT = if left.dtype.kind == dtF and left.dtype.bits == 64: types.f64 else: types.f32
  let res = applyBinary(left, right, outDT, a / b)
  if grad_enabled and (left.requires_grad or right.requires_grad):
    res.requires_grad = true
    res.creator = Op(name: "divide", inputs: @[left, right], backward: proc(g: Tensor) =
      maybe_add_grad(left, divide(g, right))
      let rightSq = multiply(right, right)
      maybe_add_grad(right, negate(multiply(g, divide(left, rightSq)))))
  res

proc `+`*(l, r: Tensor): Tensor = add(l, r)
proc `-`*(l, r: Tensor): Tensor = subtract(l, r)
proc `*`*(l, r: Tensor): Tensor = multiply(l, r)
proc `/`*(l, r: Tensor): Tensor = divide(l, r)

proc equal*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a == b: 1.0 else: 0.0))

# --- Internal Reduction ---

template reduce_loop(tIn: Tensor, ax: seq[int], kd: bool, init: float64, resOp: untyped): untyped =
  let rnk = tIn.shape.len; var t_axes = if ax.len == 0: (0..<rnk).toSeq else: ax
  for i in 0..<t_axes.len: (if t_axes[i] < 0: t_axes[i] += rnk)
  let axesSet = t_axes.toHashSet()
  var resShp: seq[int] = @[]
  for i in 0..<rnk: (if i in axesSet: (if kd: resShp.add(1)) else: resShp.add(tIn.shape[i]))
  let resT = creation.full(resShp, init, tIn.dtype)
  let sz = types.size(tIn.shape)
  if sz > 0:
    withDataPtr(tIn, pI):
      withDataPtr(resT, pO):
        for i in 0..<sz:
          var rem = i; var outIdx = 0; var outStr = 1; var kIdx = resShp.len - 1
          for j in countdown(rnk-1, 0):
            let sub = rem mod tIn.shape[j]; rem = rem div tIn.shape[j]
            if j notin axesSet: (outIdx += sub * outStr; outStr *= (if kIdx >= 0: resShp[kIdx] else: 1); kIdx -= 1)
            elif kd: (outStr *= 1; kIdx -= 1)
          let a {.inject.} = pO[outIdx].float64; let b {.inject.} = pI[i].float64
          pO[outIdx] = typeof(pO[0])(resOp)
  resT

# --- Reductions ---

proc sum*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor =
  let res = reduce_loop(tensor, axes, keep_dims, 0.0, a + b)
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "sum", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, broadcast_to(g, tensor.shape)))
  res

proc mean*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor =
  let s = sum(tensor, axes, keep_dims); var count = 1.0; let rnk = tensor.shape.len; let t_axes = if axes.len == 0: (0..<rnk).toSeq else: axes
  for ax in t_axes: (var a = ax; if a < 0: a += rnk; count *= tensor.shape[a].float64)
  let res = divide(s, creation.tensor([count.float32]).reshape(@[]))
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "mean", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, divide(broadcast_to(g, tensor.shape), creation.full(tensor.shape, count, tensor.dtype))))
  res

proc all*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor =
  reduce_loop(tensor, axes, keep_dims, 1.0, (if a != 0.0 and b != 0.0: 1.0 else: 0.0)).as_type(types.u8)

proc any*(tensor: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor =
  reduce_loop(tensor, axes, keep_dims, 0.0, (if a != 0.0 or b != 0.0: 1.0 else: 0.0)).as_type(types.u8)

# --- Shape ---

proc reshape*(tensor: Tensor, newShape: seq[int]): Tensor =
  let res = nshape.reshape(tensor, newShape)
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "reshape", inputs: @[tensor], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, reshape(g, tensor.shape)))
  res

proc broadcast_to*(tensor: Tensor, newShape: seq[int]): Tensor =
  if tensor.shape == newShape: return tensor
  let res = nshape.broadcast_to(tensor, newShape)
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "broadcast_to", inputs: @[tensor], backward: proc(g: Tensor) =
      var axes: seq[int] = @[]
      let diff = newShape.len - tensor.shape.len
      for i in 0..<diff: axes.add(i)
      for i in 0..<tensor.shape.len:
        if tensor.shape[i] == 1 and newShape[i+diff] != 1: axes.add(i+diff)
      maybe_add_grad(tensor, reshape(sum(g, axes, keep_dims=true), tensor.shape)))
  res

proc transpose*(tensor: Tensor, axes: seq[int] = @[]): Tensor =
  var ax = axes; let rnk = tensor.shape.len; if ax.len == 0: ax = (0..<rnk).toSeq.reversed()
  var nShp = newSeq[int](rnk); for i in 0..<rnk: nShp[i] = tensor.shape[ax[i]]
  let res = creation.zeros(nShp, tensor.dtype); let sz = types.size(tensor.shape)
  if sz > 0:
    withDataPtr(tensor, pI):
      withDataPtr(res, pO):
        for i in 0..<sz:
          var co = newSeq[int](rnk); var tmp = i; var str = sz
          for j in 0..<rnk: str = str div tensor.shape[j]; co[j] = tmp div str; tmp = tmp mod str
          var nCo = newSeq[int](rnk); for j in 0..<rnk: nCo[j] = co[ax[j]]
          var nIdx = 0; var nStr = 1; for j in countdown(rnk-1, 0): nIdx += nCo[j] * nStr; nStr *= nShp[j]
          pO[nIdx] = typeof(pO[0])(pI[i].float64)
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "transpose", inputs: @[tensor], backward: proc(g: Tensor) =
      var invAx = newSeq[int](rnk)
      for i in 0..<rnk: invAx[ax[i]] = i
      maybe_add_grad(tensor, transpose(g, invAx)))
  res

# --- Specialized ---

proc dot*(left, right: Tensor): Tensor =
  let lShp = left.shape; let rShp = right.shape
  if lShp.len == 0 or rShp.len == 0: return multiply(left, right)
  if lShp[^1] != rShp[0]: raise newException(ValueError, "dot shape mismatch")

  if lShp.len == 2 and rShp.len == 2:
    let res = creation.zeros(@[lShp[0], rShp[1]], left.dtype)
    withDataPtr(left, pL):
      withDataPtr(right, pR):
        withDataPtr(res, pO):
          for i in 0..<lShp[0]:
            for j in 0..<rShp[1]:
              var s: float64 = 0
              for k in 0..<lShp[1]: s += pL[i * lShp[1] + k].float64 * pR[k * rShp[1] + j].float64
              pO[i * rShp[1] + j] = typeof(pO[0])(s)
    if grad_enabled and (left.requires_grad or right.requires_grad):
      res.requires_grad = true
      res.creator = Op(name: "dot", inputs: @[left, right], backward: proc(g: Tensor) =
        maybe_add_grad(left, dot(g, transpose(right)))
        maybe_add_grad(right, dot(transpose(left), g)))
    return res
  elif lShp.len == 1 and rShp.len == 1:
    var s: float64 = 0
    withDataPtr(left, pL):
      withDataPtr(right, pR):
        for i in 0..<lShp[0]: s += pL[i].float64 * pR[i].float64
    let res = if left.dtype == types.f32: creation.tensor([s.float32]).reshape(@[]) else: creation.tensor([s]).reshape(@[])
    if grad_enabled and (left.requires_grad or right.requires_grad):
      res.requires_grad = true
      res.creator = Op(name: "dot_vv", inputs: @[left, right], backward: proc(g: Tensor) =
        maybe_add_grad(left, multiply(g, right))
        maybe_add_grad(right, multiply(g, left)))
    return res
  else:
    # General case: contract last dim of left with first dim of right
    let commonDim = lShp[^1]
    let lOuterShp = lShp[0..^2]; let rOuterShp = rShp[1..^1]
    let lOuterSize = types.size(lOuterShp); let rOuterSize = types.size(rOuterShp)
    let l2d = reshape(left, @[lOuterSize, commonDim])
    let r2d = reshape(right, @[commonDim, rOuterSize])
    let res2d = dot(l2d, r2d)
    return reshape(res2d, lOuterShp & rOuterShp)

proc take*(tensor, indices: Tensor, axis: int = 0): Tensor =
  let nInd = types.size(indices.shape); var oShp = tensor.shape; oShp[axis] = nInd
  let res = creation.zeros(oShp, tensor.dtype)
  let inSz = if axis < tensor.shape.len - 1: types.size(tensor.shape[(axis+1)..^1]) else: 1
  let outSz = if axis > 0: types.size(tensor.shape[0..<axis]) else: 1
  let axSz = tensor.shape[axis]
  withDataPtr(indices, pInd):
    withDataPtr(tensor, pIn):
      withDataPtr(res, pOut):
        for i in 0..<outSz:
          for j in 0..<nInd:
            let idx = pInd[j].int; if idx >= 0 and idx < axSz:
              for k in 0..<inSz: pOut[(i*nInd+j)*inSz+k] = typeof(pOut[0])(pIn[(i*axSz+idx)*inSz+k].float64)
  if grad_enabled and tensor.requires_grad:
    res.requires_grad = true
    res.creator = Op(name: "take", inputs: @[tensor, indices], backward: proc(g: Tensor) =
      maybe_add_grad(tensor, indexed_add(zeros_like(tensor), indices, g)))
  res

proc backward*(tensor: Tensor) =
  var sortedNodes: seq[Tensor] = @[]; var visited = initHashSet[pointer]()
  proc visit(n: Tensor) =
    if cast[pointer](n) notin visited:
      visited.incl(cast[pointer](n))
      if n.creator != nil:
        for input in n.creator.inputs: visit(input)
      sortedNodes.add(n)
  visit(tensor); algorithm.reverse(sortedNodes)
  if tensor.grad == nil: tensor.grad = creation.ones(tensor.shape, tensor.dtype)
  let old_ge = grad_enabled; grad_enabled = false
  for n in sortedNodes:
    if n.creator != nil and n.grad != nil: n.creator.backward(n.grad)
  grad_enabled = old_ge

proc grad*(f: proc(t: Tensor): Tensor): proc(t: Tensor): Tensor =
  return proc(t: Tensor): Tensor =
    let old_ge = grad_enabled; grad_enabled = true; t.requires_grad = true; t.grad = nil
    let res = f(t); backward(res); grad_enabled = old_ge; return t.grad

proc bitwise_not*(tensor: Tensor): Tensor =
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, tensor.dtype)
  if sz > 0:
    let bI = BinaryBackend(tensor.data); let bO = BinaryBackend(res.data)
    if tensor.dtype == types.s32:
      let pI = cast[ptr UncheckedArray[int32]](addr bI.buffer[0]); let pO = cast[ptr UncheckedArray[int32]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = not pI[i]
    elif tensor.dtype == types.s64:
      let pI = cast[ptr UncheckedArray[int64]](addr bI.buffer[0]); let pO = cast[ptr UncheckedArray[int64]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = not pI[i]
    elif tensor.dtype == types.u8:
      let pI = cast[ptr UncheckedArray[uint8]](addr bI.buffer[0]); let pO = cast[ptr UncheckedArray[uint8]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = not pI[i]
  res

proc bitwise_and*(l, r: Tensor): Tensor =
  let bShp = nshape.broadcast_shape(l.shape, r.shape); let lb = broadcast_to(l, bShp); let rb = broadcast_to(r, bShp)
  let res = creation.zeros(bShp, l.dtype); let sz = types.size(bShp)
  if sz > 0:
    let bL = BinaryBackend(lb.data); let bR = BinaryBackend(rb.data); let bO = BinaryBackend(res.data)
    if lb.dtype == types.s32:
      let pL = cast[ptr UncheckedArray[int32]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int32]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int32]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] and pR[i]
    elif lb.dtype == types.s64:
      let pL = cast[ptr UncheckedArray[int64]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int64]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int64]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] and pR[i]
    elif lb.dtype == types.u8:
      let pL = cast[ptr UncheckedArray[uint8]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[uint8]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[uint8]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] and pR[i]
  res

proc bitwise_or*(l, r: Tensor): Tensor =
  let bShp = nshape.broadcast_shape(l.shape, r.shape); let lb = broadcast_to(l, bShp); let rb = broadcast_to(r, bShp)
  let res = creation.zeros(bShp, l.dtype); let sz = types.size(bShp)
  if sz > 0:
    let bL = BinaryBackend(lb.data); let bR = BinaryBackend(rb.data); let bO = BinaryBackend(res.data)
    if lb.dtype == types.s32:
      let pL = cast[ptr UncheckedArray[int32]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int32]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int32]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] or pR[i]
    elif lb.dtype == types.s64:
      let pL = cast[ptr UncheckedArray[int64]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int64]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int64]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] or pR[i]
    elif lb.dtype == types.u8:
      let pL = cast[ptr UncheckedArray[uint8]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[uint8]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[uint8]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] or pR[i]
  res

proc bitwise_xor*(l, r: Tensor): Tensor =
  let bShp = nshape.broadcast_shape(l.shape, r.shape); let lb = broadcast_to(l, bShp); let rb = broadcast_to(r, bShp)
  let res = creation.zeros(bShp, l.dtype); let sz = types.size(bShp)
  if sz > 0:
    let bL = BinaryBackend(lb.data); let bR = BinaryBackend(rb.data); let bO = BinaryBackend(res.data)
    if lb.dtype == types.s32:
      let pL = cast[ptr UncheckedArray[int32]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int32]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int32]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] xor pR[i]
    elif lb.dtype == types.s64:
      let pL = cast[ptr UncheckedArray[int64]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int64]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int64]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] xor pR[i]
    elif lb.dtype == types.u8:
      let pL = cast[ptr UncheckedArray[uint8]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[uint8]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[uint8]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] xor pR[i]
  res

proc left_shift*(l, r: Tensor): Tensor =
  let bShp = nshape.broadcast_shape(l.shape, r.shape); let lb = broadcast_to(l, bShp); let rb = broadcast_to(r, bShp)
  let res = creation.zeros(bShp, l.dtype); let sz = types.size(bShp)
  if sz > 0:
    let bL = BinaryBackend(lb.data); let bR = BinaryBackend(rb.data); let bO = BinaryBackend(res.data)
    if lb.dtype == types.s32:
      let pL = cast[ptr UncheckedArray[int32]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int32]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int32]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] shl pR[i]
    elif lb.dtype == types.s64:
      let pL = cast[ptr UncheckedArray[int64]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int64]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int64]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] shl pR[i]
    elif lb.dtype == types.u8:
      let pL = cast[ptr UncheckedArray[uint8]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[uint8]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[uint8]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] shl pR[i]
  res

proc right_shift*(l, r: Tensor): Tensor =
  let bShp = nshape.broadcast_shape(l.shape, r.shape); let lb = broadcast_to(l, bShp); let rb = broadcast_to(r, bShp)
  let res = creation.zeros(bShp, l.dtype); let sz = types.size(bShp)
  if sz > 0:
    let bL = BinaryBackend(lb.data); let bR = BinaryBackend(rb.data); let bO = BinaryBackend(res.data)
    if lb.dtype == types.s32:
      let pL = cast[ptr UncheckedArray[int32]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int32]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int32]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] shr pR[i]
    elif lb.dtype == types.s64:
      let pL = cast[ptr UncheckedArray[int64]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[int64]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[int64]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] shr pR[i]
    elif lb.dtype == types.u8:
      let pL = cast[ptr UncheckedArray[uint8]](addr bL.buffer[0]); let pR = cast[ptr UncheckedArray[uint8]](addr bR.buffer[0]); let pO = cast[ptr UncheckedArray[uint8]](addr bO.buffer[0])
      for i in 0..<sz: pO[i] = pL[i] shr pR[i]
  res

proc ceil*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.ceil(val))
proc floor*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.floor(val))
proc round*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.round(val))
proc sign*(tensor: Tensor): Tensor = applyUnary(tensor, tensor.dtype, (if val > 0: 1.0 elif val < 0: -1.0 else: 0.0))
proc greater*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a > b: 1.0 else: 0.0))
proc less*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a < b: 1.0 else: 0.0))
proc greater_equal*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a >= b: 1.0 else: 0.0))
proc less_equal*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a <= b: 1.0 else: 0.0))
proc not_equal*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a != b: 1.0 else: 0.0))
proc logical_and*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a != 0 and b != 0: 1.0 else: 0.0))
proc logical_or*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if a != 0 or b != 0: 1.0 else: 0.0))
proc logical_xor*(left, right: Tensor): Tensor = applyBinary(left, right, types.u8, (if (a != 0) != (b != 0): 1.0 else: 0.0))
proc expm1*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.exp(val) - 1.0)
proc log1p*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.ln(1.0 + val))
proc log2*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.log2(val))
proc log10*(tensor: Tensor): Tensor = applyUnary(tensor, types.f32, nmath.log10(val))
proc is_nan*(tensor: Tensor): Tensor = applyUnary(tensor, types.u8, (if nmath.classify(val) == fcNaN: 1.0 else: 0.0))
proc is_infinity*(tensor: Tensor): Tensor = applyUnary(tensor, types.u8, (if nmath.classify(val) in {fcInf, fcNegInf}: 1.0 else: 0.0))
proc pad*(tensor, pad_value: Tensor, padding_config: seq[(int, int, int)]): Tensor =
  var nShp = tensor.shape; for i in 0..<tensor.shape.len: (let (l, h, it) = padding_config[i]; nShp[i] = l+h+tensor.shape[i]+(tensor.shape[i]-1)*it)
  let res = creation.full(nShp, pad_value.to_number()); var st = newSeq[int](tensor.shape.len); for i in 0..<tensor.shape.len: st[i] = padding_config[i][0]
  return nshape.put_slice(res, st, tensor)
proc pad*(tensor: Tensor, pad_value: float32, padding_config: seq[(int, int, int)]): Tensor = pad(tensor, creation.tensor([pad_value]).reshape(@[]), padding_config)

proc window_sum*(tensor: Tensor, window_shape: seq[int], strides: seq[int] = @[], padding: string = "VALID"): Tensor =
  let rnk = tensor.shape.len; if rnk < 3: return tensor
  let inCh = tensor.shape[1]; let spRnk = rnk-2; let sStr = if strides.len==0: newSeqWith(spRnk, 1) else: strides
  var oSpShp = newSeq[int](spRnk); var pSt = newSeq[int](spRnk)
  for i in 0..<spRnk:
    if padding == "VALID": (oSpShp[i] = (tensor.shape[i+2]-window_shape[i]) div sStr[i] + 1; pSt[i] = 0)
    else: (oSpShp[i] = (tensor.shape[i+2]+sStr[i]-1) div sStr[i]; pSt[i] = ((oSpShp[i]-1)*sStr[i]+window_shape[i]-tensor.shape[i+2]) div 2)
  var oShp = @[tensor.shape[0], inCh]; for s in oSpShp: oShp.add(s)
  let res = creation.zeros(oShp, tensor.dtype)
  let bt = tensor.shape[0]; var inSpSz = 1; for i in 2..<rnk: inSpSz *= tensor.shape[i]
  var oSpSz = 1; for i in 0..<spRnk: oSpSz *= oSpShp[i]
  var wSize = 1; for s in window_shape: wSize *= s
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for b in 0..<bt:
        for c in 0..<inCh:
          for osi in 0..<oSpSz:
            var oCo = newSeq[int](spRnk); var tmp = osi; var str = oSpSz
            for i in 0..<spRnk: str = str div oSpShp[i]; oCo[i] = tmp div str; tmp = tmp mod str
            var sm: float64 = 0
            for wsi in 0..<wSize:
              var wCo = newSeq[int](spRnk); var tw = wsi; var sw = wSize
              for i in 0..<spRnk: sw = sw div window_shape[i]; wCo[i] = tw div sw; tw = tw mod sw
              var inCo = newSeq[int](spRnk); var inB = true
              for i in 0..<spRnk:
                inCo[i] = oCo[i]*sStr[i]+wCo[i]-pSt[i]
                if inCo[i]<0 or inCo[i]>=tensor.shape[i+2]: (inB=false; break)
              if inB:
                var inSpOff = 0; var inM = inSpSz
                for i in 0..<spRnk: (inM = inM div tensor.shape[i+2]; inSpOff += inCo[i]*inM)
                sm += pI[(b*inCh+c)*inSpSz+inSpOff].float64
            pO[(b*inCh+c)*oSpSz+osi] = typeof(pO[0])(sm)
  res

proc window_max*(tensor: Tensor, window_shape: seq[int], strides: seq[int] = @[], padding: string = "VALID"): Tensor =
  let rnk = tensor.shape.len; if rnk < 3: return tensor
  let inCh = tensor.shape[1]; let spRnk = rnk-2; let sStr = if strides.len==0: newSeqWith(spRnk, 1) else: strides
  var oSpShp = newSeq[int](spRnk); var pSt = newSeq[int](spRnk)
  for i in 0..<spRnk:
    if padding == "VALID": (oSpShp[i] = (tensor.shape[i+2]-window_shape[i]) div sStr[i] + 1; pSt[i] = 0)
    else: (oSpShp[i] = (tensor.shape[i+2]+sStr[i]-1) div sStr[i]; pSt[i] = ((oSpShp[i]-1)*sStr[i]+window_shape[i]-tensor.shape[i+2]) div 2)
  var oShp = @[tensor.shape[0], inCh]; for s in oSpShp: oShp.add(s)
  let res = creation.zeros(oShp, tensor.dtype)
  let bt = tensor.shape[0]; var inSpSz = 1; for i in 2..<rnk: inSpSz *= tensor.shape[i]
  var oSpSz = 1; for i in 0..<spRnk: oSpSz *= oSpShp[i]
  var wSize = 1; for s in window_shape: wSize *= s
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for b in 0..<bt:
        for c in 0..<inCh:
          for osi in 0..<oSpSz:
            var oCo = newSeq[int](spRnk); var tmp = osi; var str = oSpSz
            for i in 0..<spRnk: str = str div oSpShp[i]; oCo[i] = tmp div str; tmp = tmp mod str
            var mx: float64 = -Inf
            for wsi in 0..<wSize:
              var wCo = newSeq[int](spRnk); var tw = wsi; var sw = wSize
              for i in 0..<spRnk: sw = sw div window_shape[i]; wCo[i] = tw div sw; tw = tw mod sw
              var inCo = newSeq[int](spRnk); var inB = true
              for i in 0..<spRnk:
                inCo[i] = oCo[i]*sStr[i]+wCo[i]-pSt[i]
                if inCo[i]<0 or inCo[i]>=tensor.shape[i+2]: (inB=false; break)
              if inB:
                var inSpOff = 0; var inM = inSpSz
                for i in 0..<spRnk: (inM = inM div tensor.shape[i+2]; inSpOff += inCo[i]*inM)
                mx = system.max(mx, pI[(b*inCh+c)*inSpSz+inSpOff].float64)
            pO[(b*inCh+c)*oSpSz+osi] = typeof(pO[0])(mx)
  res

proc window_min*(tensor: Tensor, window_shape: seq[int], strides: seq[int] = @[], padding: string = "VALID"): Tensor =
  let rnk = tensor.shape.len; if rnk < 3: return tensor
  let inCh = tensor.shape[1]; let spRnk = rnk-2; let sStr = if strides.len==0: newSeqWith(spRnk, 1) else: strides
  var oSpShp = newSeq[int](spRnk); var pSt = newSeq[int](spRnk)
  for i in 0..<spRnk:
    if padding == "VALID": (oSpShp[i] = (tensor.shape[i+2]-window_shape[i]) div sStr[i] + 1; pSt[i] = 0)
    else: (oSpShp[i] = (tensor.shape[i+2]+sStr[i]-1) div sStr[i]; pSt[i] = ((oSpShp[i]-1)*sStr[i]+window_shape[i]-tensor.shape[i+2]) div 2)
  var oShp = @[tensor.shape[0], inCh]; for s in oSpShp: oShp.add(s)
  let res = creation.zeros(oShp, tensor.dtype)
  let bt = tensor.shape[0]; var inSpSz = 1; for i in 2..<rnk: inSpSz *= tensor.shape[i]
  var oSpSz = 1; for i in 0..<spRnk: oSpSz *= oSpShp[i]
  var wSize = 1; for s in window_shape: wSize *= s
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for b in 0..<bt:
        for c in 0..<inCh:
          for osi in 0..<oSpSz:
            var oCo = newSeq[int](spRnk); var tmp = osi; var str = oSpSz
            for i in 0..<spRnk: str = str div oSpShp[i]; oCo[i] = tmp div str; tmp = tmp mod str
            var mn: float64 = Inf
            for wsi in 0..<wSize:
              var wCo = newSeq[int](spRnk); var tw = wsi; var sw = wSize
              for i in 0..<spRnk: sw = sw div window_shape[i]; wCo[i] = tw div sw; tw = tw mod sw
              var inCo = newSeq[int](spRnk); var inB = true
              for i in 0..<spRnk:
                inCo[i] = oCo[i]*sStr[i]+wCo[i]-pSt[i]
                if inCo[i]<0 or inCo[i]>=tensor.shape[i+2]: (inB=false; break)
              if inB:
                var inSpOff = 0; var inM = inSpSz
                for i in 0..<spRnk: (inM = inM div tensor.shape[i+2]; inSpOff += inCo[i]*inM)
                mn = system.min(mn, pI[(b*inCh+c)*inSpSz+inSpOff].float64)
            pO[(b*inCh+c)*oSpSz+osi] = typeof(pO[0])(mn)
  res

proc window_product*(tensor: Tensor, window_shape: seq[int], strides: seq[int] = @[], padding: string = "VALID"): Tensor =
  let rnk = tensor.shape.len; if rnk < 3: return tensor
  let inCh = tensor.shape[1]; let spRnk = rnk-2; let sStr = if strides.len==0: newSeqWith(spRnk, 1) else: strides
  var oSpShp = newSeq[int](spRnk); var pSt = newSeq[int](spRnk)
  for i in 0..<spRnk:
    if padding == "VALID": (oSpShp[i] = (tensor.shape[i+2]-window_shape[i]) div sStr[i] + 1; pSt[i] = 0)
    else: (oSpShp[i] = (tensor.shape[i+2]+sStr[i]-1) div sStr[i]; pSt[i] = ((oSpShp[i]-1)*sStr[i]+window_shape[i]-tensor.shape[i+2]) div 2)
  var oShp = @[tensor.shape[0], inCh]; for s in oSpShp: oShp.add(s)
  let res = creation.zeros(oShp, tensor.dtype)
  let bt = tensor.shape[0]; var inSpSz = 1; for i in 2..<rnk: inSpSz *= tensor.shape[i]
  var oSpSz = 1; for i in 0..<spRnk: oSpSz *= oSpShp[i]
  var wSize = 1; for s in window_shape: wSize *= s
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for b in 0..<bt:
        for c in 0..<inCh:
          for osi in 0..<oSpSz:
            var oCo = newSeq[int](spRnk); var tmp = osi; var str = oSpSz
            for i in 0..<spRnk: str = str div oSpShp[i]; oCo[i] = tmp div str; tmp = tmp mod str
            var pr: float64 = 1
            for wsi in 0..<wSize:
              var wCo = newSeq[int](spRnk); var tw = wsi; var sw = wSize
              for i in 0..<spRnk: sw = sw div window_shape[i]; wCo[i] = tw div sw; tw = tw mod sw
              var inCo = newSeq[int](spRnk); var inB = true
              for i in 0..<spRnk:
                inCo[i] = oCo[i]*sStr[i]+wCo[i]-pSt[i]
                if inCo[i]<0 or inCo[i]>=tensor.shape[i+2]: (inB=false; break)
              if inB:
                var inSpOff = 0; var inM = inSpSz
                for i in 0..<spRnk: (inM = inM div tensor.shape[i+2]; inSpOff += inCo[i]*inM)
                pr *= pI[(b*inCh+c)*inSpSz+inSpOff].float64
            pO[(b*inCh+c)*oSpSz+osi] = typeof(pO[0])(pr)
  res

proc window_mean*(tensor: Tensor, window_shape: seq[int], strides: seq[int] = @[], padding: string = "VALID"): Tensor =
  let ws = window_sum(tensor, window_shape, strides, padding)
  var wSize = 1.0; for s in window_shape: wSize *= s.float64
  divide(ws, creation.tensor([wSize.float32]).reshape(@[]))

proc indexed_add*(t, indices, updates: Tensor): Tensor =
  let res = creation.zeros(t.shape, t.dtype); let tSz = types.size(t.shape)
  withDataPtr(t, pT): withDataPtr(res, pR): (for i in 0..<tSz: pR[i] = typeof(pR[0])(pT[i].float64))
  withDataPtr(indices, pInd):
    withDataPtr(updates, pUpd):
      withDataPtr(res, pR):
        for i in 0..<indices.shape[0]:
          var fV = 0; var str = 1; for j in countdown(t.shape.len-1, 0): (var c = 0; if j < indices.shape[1]: c = pInd[i*indices.shape[1]+j].int; fV += c * str; str *= t.shape[j])
          if fV >= 0 and fV < tSz: pR[fV] = typeof(pR[0])(pR[fV].float64 + pUpd[i].float64)
  res

proc indexed_put*(t, indices, updates: Tensor): Tensor =
  let res = creation.zeros(t.shape, t.dtype); let tSz = types.size(t.shape)
  withDataPtr(t, pT): withDataPtr(res, pR): (for i in 0..<tSz: pR[i] = typeof(pR[0])(pT[i].float64))
  withDataPtr(indices, pInd):
    withDataPtr(updates, pUpd):
      withDataPtr(res, pR):
        for i in 0..<indices.shape[0]:
          var fV = 0; var str = 1; for j in countdown(t.shape.len-1, 0): (var c = 0; if j < indices.shape[1]: c = pInd[i*indices.shape[1]+j].int; fV += c * str; str *= t.shape[j])
          if fV >= 0 and fV < tSz: pR[fV] = typeof(pR[0])(pUpd[i].float64)
  res

proc conv*(input, kernel: Tensor, strides: seq[int] = @[], padding: string = "VALID"): Tensor =
  let iShp = input.shape; let kShp = kernel.shape; let rnk = iShp.len
  if rnk < 3: return input
  let bt = iShp[0]; let inCh = iShp[1]; let oCh = kShp[0]; let spRnk = rnk-2
  let sStr = if strides.len==0: newSeqWith(spRnk, 1) else: strides
  var oSpShp = newSeq[int](spRnk); var pSt = newSeq[int](spRnk)
  for i in 0..<spRnk:
    if padding == "VALID": (oSpShp[i] = (iShp[i+2]-kShp[i+2]) div sStr[i] + 1; pSt[i] = 0)
    else: (oSpShp[i] = (iShp[i+2]+sStr[i]-1) div sStr[i]; pSt[i] = ((oSpShp[i]-1)*sStr[i]+kShp[i+2]-iShp[i+2]) div 2)
  var oShp = @[bt, oCh]; for s in oSpShp: oShp.add(s)
  let res = creation.zeros(oShp, input.dtype)
  var kSize = 1; for i in 2..<rnk: kSize *= kShp[i]
  var oSpSz = 1; for i in 0..<spRnk: oSpSz *= oSpShp[i]
  var iSpSz = 1; for i in 2..<rnk: iSpSz *= iShp[i]
  withDataPtr(input, pI):
    withDataPtr(kernel, pK):
      withDataPtr(res, pO):
        for b in 0..<bt:
          for oc in 0..<oCh:
            for osi in 0..<oSpSz:
              var oCo = newSeq[int](spRnk); var tmp = osi; var str = oSpSz
              for i in 0..<spRnk: str = str div oSpShp[i]; oCo[i] = tmp div str; tmp = tmp mod str
              var sm: float64 = 0
              for ic in 0..<inCh:
                for ksi in 0..<kSize:
                  var kCo = newSeq[int](spRnk); var tk = ksi; var sk = kSize
                  for i in 0..<spRnk: sk = sk div kShp[i+2]; kCo[i] = tk div sk; tk = tk mod sk
                  var inCo = newSeq[int](spRnk); var inB = true
                  for i in 0..<spRnk:
                    inCo[i] = oCo[i]*sStr[i]+kCo[i]-pSt[i]
                    if inCo[i]<0 or inCo[i]>=iShp[i+2]: (inB=false; break)
                  if inB:
                    var inSpOff = 0; var inM = iSpSz
                    for i in 0..<spRnk: (inM = inM div iShp[i+2]; inSpOff += inCo[i]*inM)
                    sm += pI[(b*inCh+ic)*iSpSz+inSpOff].float64 * pK[(oc*inCh+ic)*kSize+ksi].float64
              pO[(b*oCh+oc)*oSpSz+osi] = typeof(pO[0])(sm)
  res

proc sort*(tensor: Tensor, axis: int = -1): Tensor =
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, tensor.dtype)
  var d = newSeq[float64](sz)
  withDataPtr(tensor, p): (for i in 0..<sz: d[i] = p[i].float64)
  d.sort()
  withDataPtr(res, pO): (for i in 0..<sz: pO[i] = typeof(pO[0])(d[i]))
  res

proc population_count*(tensor: Tensor): Tensor =
  applyUnary(tensor, types.s32, (let v = val.uint64; bitops.countSetBits(v).float64))

proc count_leading_zeros*(tensor: Tensor): Tensor =
  applyUnary(tensor, types.s32, (let v = val.uint64; bitops.countLeadingZeroBits(v).float64))

proc cumulative_sum*(tensor: Tensor, axis: int = 0): Tensor =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, tensor.dtype)
  let axisDim = tensor.shape[ax]
  let innerSize = if ax == rnk - 1: 1 else: types.size(tensor.shape[(ax+1)..^1])
  let outerSize = sz div (axisDim * innerSize)
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for o in 0..<outerSize:
        for inIdx in 0..<innerSize:
          var acc: float64 = 0
          for i in 0..<axisDim:
            let idx = (o * axisDim + i) * innerSize + inIdx
            acc += pI[idx].float64; pO[idx] = typeof(pO[0])(acc)
  res

proc cumulative_product*(tensor: Tensor, axis: int = 0): Tensor =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, tensor.dtype)
  let axisDim = tensor.shape[ax]
  let innerSize = if ax == rnk - 1: 1 else: types.size(tensor.shape[(ax+1)..^1])
  let outerSize = sz div (axisDim * innerSize)
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for o in 0..<outerSize:
        for inIdx in 0..<innerSize:
          var acc: float64 = 1
          for i in 0..<axisDim:
            let idx = (o * axisDim + i) * innerSize + inIdx
            acc *= pI[idx].float64; pO[idx] = typeof(pO[0])(acc)
  res

proc cumulative_max*(tensor: Tensor, axis: int = 0): Tensor =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, tensor.dtype)
  let axisDim = tensor.shape[ax]
  let innerSize = if ax == rnk - 1: 1 else: types.size(tensor.shape[(ax+1)..^1])
  let outerSize = sz div (axisDim * innerSize)
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for o in 0..<outerSize:
        for inIdx in 0..<innerSize:
          var acc: float64 = -Inf
          for i in 0..<axisDim:
            let idx = (o * axisDim + i) * innerSize + inIdx
            acc = system.max(acc, pI[idx].float64); pO[idx] = typeof(pO[0])(acc)
  res

proc cumulative_min*(tensor: Tensor, axis: int = 0): Tensor =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let sz = types.size(tensor.shape); let res = creation.zeros(tensor.shape, tensor.dtype)
  let axisDim = tensor.shape[ax]
  let innerSize = if ax == rnk - 1: 1 else: types.size(tensor.shape[(ax+1)..^1])
  let outerSize = sz div (axisDim * innerSize)
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for o in 0..<outerSize:
        for inIdx in 0..<innerSize:
          var acc: float64 = Inf
          for i in 0..<axisDim:
            let idx = (o * axisDim + i) * innerSize + inIdx
            acc = system.min(acc, pI[idx].float64); pO[idx] = typeof(pO[0])(acc)
  res

proc outer*(left, right: Tensor): Tensor =
  let a = nshape.flatten(left); let b = nshape.flatten(right)
  let res = creation.zeros(@[a.shape[0], b.shape[0]], left.dtype)
  withDataPtr(a, pL):
    withDataPtr(b, pR):
      withDataPtr(res, pO):
        for i in 0..<a.shape[0]:
          for j in 0..<b.shape[0]:
            pO[i * b.shape[0] + j] = typeof(pO[0])(pL[i].float64 * pR[j].float64)
  if grad_enabled and (left.requires_grad or right.requires_grad):
    res.requires_grad = true
    res.creator = Op(name: "outer", inputs: @[left, right], backward: proc(g: Tensor) =
      maybe_add_grad(left, reshape(dot(g, b), left.shape))
      maybe_add_grad(right, reshape(dot(transpose(a), g), right.shape)))
  res

proc argmin*(tensor: Tensor, axis: int = -1, keep_dims: bool = false): Tensor =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let sz = types.size(tensor.shape); let axisDim = tensor.shape[ax]
  let innerSize = if ax == rnk - 1: 1 else: types.size(tensor.shape[(ax+1)..^1])
  let outerSize = sz div (axisDim * innerSize)
  var resShp: seq[int] = @[]
  for i in 0..<rnk: (if i == ax: (if keep_dims: resShp.add(1)) else: resShp.add(tensor.shape[i]))
  let res = creation.zeros(resShp, types.s32)
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for o in 0..<outerSize:
        for inIdx in 0..<innerSize:
          var minVal: float64 = Inf; var minIdx = 0
          for i in 0..<axisDim:
            let idx = (o * axisDim + i) * innerSize + inIdx
            if pI[idx].float64 < minVal: (minVal = pI[idx].float64; minIdx = i)
          pO[o * innerSize + inIdx] = typeof(pO[0])(minIdx)
  res

proc argmax*(tensor: Tensor, axis: int = -1, keep_dims: bool = false): Tensor =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let sz = types.size(tensor.shape); let axisDim = tensor.shape[ax]
  let innerSize = if ax == rnk - 1: 1 else: types.size(tensor.shape[(ax+1)..^1])
  let outerSize = sz div (axisDim * innerSize)
  var resShp: seq[int] = @[]
  for i in 0..<rnk: (if i == ax: (if keep_dims: resShp.add(1)) else: resShp.add(tensor.shape[i]))
  let res = creation.zeros(resShp, types.s32)
  withDataPtr(tensor, pI):
    withDataPtr(res, pO):
      for o in 0..<outerSize:
        for inIdx in 0..<innerSize:
          var maxVal: float64 = -Inf; var maxIdx = 0
          for i in 0..<axisDim:
            let idx = (o * axisDim + i) * innerSize + inIdx
            if pI[idx].float64 > maxVal: (maxVal = pI[idx].float64; maxIdx = i)
          pO[o * innerSize + inIdx] = typeof(pO[0])(maxIdx)
  res

proc weighted_mean*(tensor, weights: Tensor, axes: seq[int] = @[], keep_dims: bool = false): Tensor =
  let prod = multiply(tensor, weights)
  let s = sum(prod, axes, keep_dims)
  let wSum = sum(weights, axes, keep_dims)
  divide(s, wSum)

proc select*(pred, on_true, on_false: Tensor): Tensor =
  let bShp = nshape.broadcast_shape(pred.shape, nshape.broadcast_shape(on_true.shape, on_false.shape))
  let bp = broadcast_to(pred, bShp)
  let bt = broadcast_to(on_true, bShp)
  let bf = broadcast_to(on_false, bShp)
  let res = creation.zeros(bShp, bt.dtype)
  let sz = types.size(bShp)
  withDataPtr(bp, pP):
    withDataPtr(bt, pT):
      withDataPtr(bf, pF):
        withDataPtr(res, pO):
          for i in 0..<sz:
            pO[i] = typeof(pO[0])(if pP[i].float64 != 0.0: pT[i].float64 else: pF[i].float64)
  if grad_enabled and (on_true.requires_grad or on_false.requires_grad):
    res.requires_grad = true
    res.creator = Op(name: "select", inputs: @[pred, on_true, on_false], backward: proc(g: Tensor) =
      maybe_add_grad(on_true, select(pred, g, zeros_like(g)))
      maybe_add_grad(on_false, select(pred, zeros_like(g), g)))
  res

proc while_loop*(cond_fun: proc(t: Tensor): Tensor, body_fun: proc(t: Tensor): Tensor, init: Tensor): Tensor =
  var res = init
  while true:
    let c = cond_fun(res)
    if c.to_number() == 0: break
    res = body_fun(res)
  res

proc scan*(init: Tensor, tensor: Tensor, fun: proc(acc: Tensor, x: Tensor): (Tensor, Tensor), axis: int = 0): (Tensor, Tensor) =
  let rnk = tensor.shape.len; var ax = axis; if ax < 0: ax += rnk
  let axisDim = tensor.shape[ax]
  var acc = init
  var outs: seq[Tensor] = @[]
  for i in 0..<axisDim:
    let x = nshape.slice_along_axis(tensor, i, 1, ax)
    let (new_acc, out_val) = fun(acc, x.squeeze(@[ax]))
    acc = new_acc
    outs.add(out_val.reshape(x.shape))
  (acc, nshape.concatenate(outs, ax))
