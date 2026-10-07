# Project resume contract

- Rebased the local change onto the latest `origin/master` tip `0e320dc8`.
- Replaced the environment-dependent Python test with a pytest GPU integration test. It creates two tiny transforms scenes, trains the seed project, and covers compatible relocation, incompatible data rejection, sidecar masks, and embedded fallback. The final run passed in 3.14 seconds.
- The same test failed on the task-start baseline (`cb5e6b3`): resuming after relocation tried the old image paths and failed to load them.
- `GrantedTrainerSaveAtIterWithoutPathWritesBoundFile` passed three times on baseline and three times on this branch. The reported timeout did not reproduce.
- The source scan found 8 native tests referencing the resume/install symbols or `resume_project`; all were covered, along with dataset/config override tests and the requested project-checkpoint, checkpoint-resume, project-chapter, and visualizer-reset suites. The Vulkan parameterized suites were selected by their instantiated GoogleTest names.
- CUDA scope: 272 tests across 10 suites; 269 passed, 3 skipped. Vulkan scope: 272 tests across 10 suites; 269 passed, 3 skipped. No failures.
- `LichtFeld-Studio` and `lichtfeld_tests` Release targets built. `check_python_stubs` passed, and clang-format plus `git diff --check` passed.

Local work only; nothing was pushed.
