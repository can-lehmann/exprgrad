version = "0.0.1"
author = "Can Joshua Lehmann"
description = "An experimental deep learning framework"
license = "Apache License 2.0"

requires "nim >= 1.6.0"

task testOnDocker, "Run tests on docker":
  withDir "docker":
    exec "docker-compose run --rm app"
