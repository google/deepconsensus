package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "tf_example_utils",
    srcs = ["tf_example_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "tf_example_utils_test",
    srcs = ["tf_example_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_example_transforms",
        ":tf_example_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "tf_example_transforms",
    srcs = ["tf_example_transforms.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_example_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "tf_example_transforms_test",
    srcs = ["tf_example_transforms_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":tf_example_transforms",
        ":tf_example_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "write_tf_examples_lib",
    srcs = ["write_tf_examples.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_example_transforms",
        ":tf_example_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "write_tf_examples",
    srcs = ["write_tf_examples.py"],
    python_version = "PY3",
    deps = [
        ":write_tf_examples_lib",
    ],
)

py_strict_test(
    name = "write_tf_examples_test",
    srcs = ["write_tf_examples_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":tf_example_utils",
        ":write_tf_examples_lib",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
