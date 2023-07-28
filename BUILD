filegroup(
    name = "tiny_shakespeare",
    srcs = ["tiny_shakespeare.txt"],
)

cc_binary(
    name = "nano_gpt_main_bin",
    srcs = ["nano_gpt_main.cc"],
    deps = [
        ":nano_gpt_main",
    ],
)

cc_library(
    name = "nano_gpt_main",
    srcs = ["nano_gpt_main.cc"],
    data = [":tiny_shakespeare"],
    deps = [
        ":tokenizer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/flags:flag",
        "@libtorch_repo//:libtorch",
    ],
)

cc_library(
    name = "tokenizer",
    srcs = ["tokenizer.cc"],
    hdrs = ["tokenizer.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "example",
    srcs = ["example.cpp"],
    deps = [
        "@libtorch_repo//:libtorch",
    ],
)
