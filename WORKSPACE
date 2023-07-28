load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_google_absl",
    branch = "master",
    remote = "https://github.com/abseil/abseil-cpp.git",
)

git_repository(
    name = "bazel_skylib",
    branch = "main",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

git_repository(
    name = "com_google_googletest",
    branch = "main",
    remote = "https://github.com/google/googletest.git",
)

git_repository(
    name = "com_google_benchmark",
    branch = "main",
    remote = "https://github.com/google/benchmark.git",
)

git_repository(
    name = "com_google_protobuf",
    branch = "main",
    remote = "https://github.com/protocolbuffers/protobuf.git",
)

# Load common dependencies.
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Libtorch logic should be tweaked based on findings in https://github.com/uber/neuropod.git
load("//third_party/libtorch:libtorch.bzl", "libtorch_repository")

http_archive(
    name = "mklml_repo_darwin",
    build_file = "@//third_party/mklml:BUILD",
    sha256 = "2fbb71a0365d42a39ea7906568d69b1db3bfc9914fee75eedb06c5f32bf5fa68",
    strip_prefix = "mklml_mac_2019.0.5.20190502",
    url = "https://github.com/intel/mkl-dnn/releases/download/v0.19/mklml_mac_2019.0.5.20190502.tgz",
)

libtorch_repository(
    name = "libtorch_repo",
    build_file_template = "@//third_party/libtorch:BUILD",
)

git_repository(
  name = "hedron_compile_commands",
  branch = "main",
  remote = "https://github.com/hedronvision/bazel-compile-commands-extractor.git"
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()

# bazel run @hedron_compile_commands//:refresh_all every time you create a module for LSP support