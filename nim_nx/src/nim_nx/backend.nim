import types

type
  BinaryBackend* = ref object of BackendData
    buffer*: seq[byte]

proc newBinaryBackend*(size: int): BinaryBackend =
  BinaryBackend(buffer: newSeq[byte](size))

proc newBinaryBackend*(buffer: seq[byte]): BinaryBackend =
  BinaryBackend(buffer: buffer)
