import std/os, std/osproc, std/strutils

let testDir = "nim_nx/test"
var failed = false

for file in walkDirRec(testDir):
  if file.endsWith(".nim"):
    echo "Running test: ", file
    let (output, exitCode) = execCmdEx("nim r -p:nim_nx/src " & file)
    if exitCode != 0:
      echo "FAILED: ", file
      echo output
      failed = true
    else:
      echo "PASSED"

if failed:
  quit(1)
else:
  echo "ALL TESTS PASSED"
