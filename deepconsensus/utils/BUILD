package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_binary(
    name = "compare_metrics_to_pbccs",
    srcs = ["compare_metrics_to_pbccs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_library(name = "ipython")

py_library(
    name = "py_test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":ipython",  # buildcleaner: keep
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_library(
    name = "dc_constants",
    srcs = ["dc_constants.py"],
    srcs_version = "PY3",
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
