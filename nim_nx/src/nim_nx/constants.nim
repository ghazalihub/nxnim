import std/math

const
  pi* = math.PI
  e* = math.E
  euler_gamma* = 0.57721566490153286060651209008240243104215933593992

proc infinity*(): float64 = system.Inf
proc nan*(): float64 = system.NaN

proc min_finite*(): float64 = -1.7976931348623157e+308 # for f64
proc max_finite*(): float64 = 1.7976931348623157e+308
