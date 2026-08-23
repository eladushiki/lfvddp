# Issue 009: Reorganize configuration packs

**Status:** Done

## Description

Reorganize the project `configs` directory around complete packs that can be passed directly to `--configs`. Provide tracked loaded-dataset and generated-dataset packs, while allowing copied local packs to remain ignored.

## Completion

Implemented `configs/basic-loaded` and `configs/basic-generated`, each containing every required configuration fragment. Updated test defaults, documentation, and ignore rules; both pack directories are verified through the real `--configs` parser path.
