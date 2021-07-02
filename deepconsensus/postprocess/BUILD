py_library(
    name = "stitch_predictions_transforms",
    srcs = ["stitch_predictions_transforms.py"],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "stitch_predictions_transforms_test",
    srcs = ["stitch_predictions_transforms_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stitch_predictions_transforms",
        "//deepconsensus/utils",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "stitch_predictions_test",
    srcs = ["stitch_predictions_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stitch_predictions.testonly_lib",
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_binary(
    name = "stitch_predictions",
    srcs = ["stitch_predictions.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    test_lib = True,
    deps = [
        ":stitch_predictions_transforms",
        "//deepconsensus/models:model_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "plx_ingest",
    srcs = ["plx_ingest.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    test_lib = True,
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)
