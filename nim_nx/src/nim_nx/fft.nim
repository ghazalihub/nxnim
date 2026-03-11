import types, creation, backend, ops, shape
import std/math

proc fft_internal[T](data_re: var seq[T], data_im: var seq[T], inverse: bool) =
  let n = data_re.len
  if n <= 1: return

  if (n and (n - 1)) != 0:
    var res_re = newSeq[T](n); var res_im = newSeq[T](n)
    let sign = if inverse: 1.0.T else: -1.0.T
    for k in 0..<n:
      var re: T = 0; var im: T = 0
      for t in 0..<n:
        let angle = 2.0 * PI * k.float64 * t.float64 / n.float64
        let cos_a = cos(angle).T; let sin_a = (sign * sin(angle).T).T
        re += data_re[t] * cos_a - data_im[t] * sin_a
        im += data_re[t] * sin_a + data_im[t] * cos_a
      res_re[k] = re; res_im[k] = im
    if inverse:
      for i in 0..<n: (data_re[i] = res_re[i] / n.T; data_im[i] = res_im[i] / n.T)
    else:
      for i in 0..<n: (data_re[i] = res_re[i]; data_im[i] = res_im[i])
    return

  var even_re = newSeq[T](n div 2); var even_im = newSeq[T](n div 2)
  var odd_re = newSeq[T](n div 2); var odd_im = newSeq[T](n div 2)
  for i in 0 ..< n div 2:
    even_re[i] = data_re[2 * i]; even_im[i] = data_im[2 * i]
    odd_re[i] = data_re[2 * i + 1]; odd_im[i] = data_im[2 * i + 1]
  fft_internal(even_re, even_im, inverse); fft_internal(odd_re, odd_im, inverse)
  let sign = if inverse: 1.0.T else: -1.0.T
  for k in 0 ..< n div 2:
    let angle = 2.0 * PI * k.float64 / n.float64
    let w_re = cos(angle).T; let w_im = (sign * sin(angle).T).T
    let t_re = w_re * odd_re[k] - w_im * odd_im[k]
    let t_im = w_re * odd_im[k] + w_im * odd_re[k]
    data_re[k] = even_re[k] + t_re; data_im[k] = even_im[k] + t_im
    data_re[k + n div 2] = even_re[k] - t_re; data_im[k + n div 2] = even_im[k] - t_im

proc fft*(tensor: Tensor): Tensor =
  let n = tensor.shape.size()
  if tensor.dtype == c128 or tensor.dtype == f64:
    var data_re = newSeq[float64](n); var data_im = newSeq[float64](n)
    let bkd = BinaryBackend(tensor.data)
    if bkd.buffer.len > 0:
      if tensor.dtype == c128:
        let p = cast[ptr UncheckedArray[float64]](addr bkd.buffer[0])
        for i in 0..<n: (data_re[i] = p[i*2]; data_im[i] = p[i*2+1])
      else:
        let t64 = ops.as_type(tensor, f64)
        let p = cast[ptr UncheckedArray[float64]](addr BinaryBackend(t64.data).buffer[0])
        for i in 0..<n: (data_re[i] = p[i]; data_im[i] = 0)
    fft_internal(data_re, data_im, false)
    var res = zeros(tensor.shape, c128); let pD = cast[ptr UncheckedArray[float64]](addr BinaryBackend(res.data).buffer[0])
    for i in 0..<n: (pD[i*2] = data_re[i]; pD[i*2+1] = data_im[i])
    return res
  else:
    var data_re = newSeq[float32](n); var data_im = newSeq[float32](n)
    let bkd = BinaryBackend(tensor.data)
    if bkd.buffer.len > 0:
      if tensor.dtype == c64:
        let p = cast[ptr UncheckedArray[float32]](addr bkd.buffer[0])
        for i in 0..<n: (data_re[i] = p[i*2]; data_im[i] = p[i*2+1])
      else:
        let t32 = ops.as_type(tensor, f32)
        let p = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t32.data).buffer[0])
        for i in 0..<n: (data_re[i] = p[i]; data_im[i] = 0)
    fft_internal(data_re, data_im, false)
    var res = zeros(tensor.shape, c64); let pD = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
    for i in 0..<n: (pD[i*2] = data_re[i]; pD[i*2+1] = data_im[i])
    return res

proc ifft*(tensor: Tensor): Tensor =
  let n = tensor.shape.size()
  if tensor.dtype == c128 or tensor.dtype == f64:
    var data_re = newSeq[float64](n); var data_im = newSeq[float64](n)
    let bkd = BinaryBackend(tensor.data)
    if bkd.buffer.len > 0:
      if tensor.dtype == c128:
        let p = cast[ptr UncheckedArray[float64]](addr bkd.buffer[0])
        for i in 0..<n: (data_re[i] = p[i*2]; data_im[i] = p[i*2+1])
      else:
        let t64 = ops.as_type(tensor, f64)
        let p = cast[ptr UncheckedArray[float64]](addr BinaryBackend(t64.data).buffer[0])
        for i in 0..<n: (data_re[i] = p[i]; data_im[i] = 0)
    fft_internal(data_re, data_im, true)
    var res = zeros(tensor.shape, c128); let pD = cast[ptr UncheckedArray[float64]](addr BinaryBackend(res.data).buffer[0])
    for i in 0..<n: (pD[i*2] = data_re[i]; pD[i*2+1] = data_im[i])
    return res
  else:
    var data_re = newSeq[float32](n); var data_im = newSeq[float32](n)
    let bkd = BinaryBackend(tensor.data)
    if bkd.buffer.len > 0:
      if tensor.dtype == c64:
        let p = cast[ptr UncheckedArray[float32]](addr bkd.buffer[0])
        for i in 0..<n: (data_re[i] = p[i*2]; data_im[i] = p[i*2+1])
      else:
        let t32 = ops.as_type(tensor, f32)
        let p = cast[ptr UncheckedArray[float32]](addr BinaryBackend(t32.data).buffer[0])
        for i in 0..<n: (data_re[i] = p[i]; data_im[i] = 0)
    fft_internal(data_re, data_im, true)
    var res = zeros(tensor.shape, c64); let pD = cast[ptr UncheckedArray[float32]](addr BinaryBackend(res.data).buffer[0])
    for i in 0..<n: (pD[i*2] = data_re[i]; pD[i*2+1] = data_im[i])
    return res

proc fft2*(tensor: Tensor): Tensor =
  let rows = tensor.shape[0]; let cols = tensor.shape[1]
  let outDT = if tensor.dtype.kind == dtC: (if tensor.dtype.bits == 128: c128 else: c64) elif tensor.dtype.kind == dtF: (if tensor.dtype.bits == 64: c128 else: c64) else: c64
  var intermediate = zeros(@[rows, cols], outDT)
  for i in 0..<rows:
    let row = ops.reshape(slice(tensor, @[i, 0], @[1, cols]), @[cols])
    let row_fft = fft(row); intermediate = put_slice(intermediate, @[i, 0], ops.reshape(row_fft, @[1, cols]))
  var res = zeros(@[rows, cols], outDT)
  for j in 0..<cols:
    let col = ops.reshape(slice(intermediate, @[0, j], @[rows, 1]), @[rows])
    let col_fft = fft(col); res = put_slice(res, @[0, j], ops.reshape(col_fft, @[rows, 1]))
  res

proc ifft2*(tensor: Tensor): Tensor =
  let rows = tensor.shape[0]; let cols = tensor.shape[1]
  let outDT = if tensor.dtype.kind == dtC: (if tensor.dtype.bits == 128: c128 else: c64) elif tensor.dtype.kind == dtF: (if tensor.dtype.bits == 64: c128 else: c64) else: c64
  var intermediate = zeros(@[rows, cols], outDT)
  for i in 0..<rows:
    let row = ops.reshape(slice(tensor, @[i, 0], @[1, cols]), @[cols])
    let row_ifft = ifft(row); intermediate = put_slice(intermediate, @[i, 0], ops.reshape(row_ifft, @[1, cols]))
  var res = zeros(@[rows, cols], outDT)
  for j in 0..<cols:
    let col = ops.reshape(slice(intermediate, @[0, j], @[rows, 1]), @[rows])
    let col_ifft = ifft(col); res = put_slice(res, @[0, j], ops.reshape(col_ifft, @[rows, 1]))
  res
