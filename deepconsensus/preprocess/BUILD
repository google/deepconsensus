package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "preprocess_utils",
    srcs = [
        "preprocess_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
    ],
)

py_strict_test(
    name = "preprocess_utils_test",
    srcs = [
        "preprocess_utils_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":preprocess_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "beam_io",
    srcs = [
        "beam_io.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
    ],
)

py_strict_test(
    name = "beam_io_test",
    size = "small",
    srcs = ["beam_io_test.py"],
    data = ["//deepconsensus:testdata"],
    python_version = "PY3",
    deps = [
        ":beam_io",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "merge_datasets_transforms",
    srcs = [
        "merge_datasets_transforms.py",
    ],
    srcs_version = "PY3",
    deps = [":preprocess_utils"],
)

py_strict_test(
    name = "merge_datasets_transforms_test",
    srcs = ["merge_datasets_transforms_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":merge_datasets_transforms",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "merge_datasets_lib",
    srcs = [
        "merge_datasets.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":beam_io",
        ":merge_datasets_transforms",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "merge_datasets",
    srcs = ["merge_datasets.py"],
    python_version = "PY3",
    deps = [
        ":beam_io",
        ":merge_datasets_transforms",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "merge_datasets_test",
    srcs = ["merge_datasets_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":merge_datasets_lib",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "generate_input_transforms",
    srcs = ["generate_input_transforms.py"],
    srcs_version = "PY3",
    deps = [
        ":preprocess_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "generate_input_transforms_test",
    srcs = ["generate_input_transforms_test.py"],
    python_version = "PY3",
    deps = [
        ":generate_input_transforms",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "generate_input_lib",
    srcs = ["generate_input.py"],
    srcs_version = "PY3",
    deps = [
        ":beam_io",
        ":generate_input_transforms",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "generate_input",
    srcs = ["generate_input.py"],
    python_version = "PY3",
    deps = [
        ":beam_io",
        ":generate_input_transforms",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "generate_input_test",
    srcs = ["generate_input_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":generate_input_lib",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
