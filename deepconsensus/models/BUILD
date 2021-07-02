package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "majority_vote_transforms",
    srcs = [
        "majority_vote_transforms.py",
    ],
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "majority_vote_transforms_test",
    srcs = ["majority_vote_transforms_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":majority_vote_transforms",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "run_majority_vote_model_lib",
    srcs = [
        "run_majority_vote_model.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":majority_vote_transforms",
        ":model_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/tf_examples:tf_example_transforms",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "run_majority_vote_model",
    srcs = ["run_majority_vote_model.py"],
    python_version = "PY3",
    deps = [
        ":majority_vote_transforms",
        ":model_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/tf_examples:tf_example_transforms",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "run_majority_vote_model_test",
    srcs = ["run_majority_vote_model_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":run_majority_vote_model_lib",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "data_providers",
    srcs = ["data_providers.py"],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/tf_examples:tf_example_utils",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "data_providers_test",
    timeout = "long",
    srcs = ["data_providers_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":data_providers",
        ":model_configs",
        ":model_utils",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/tf_examples:tf_example_utils",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "networks",
    srcs = ["networks.py"],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/tf_examples:tf_example_utils",
    ],
)

py_strict_test(
    name = "networks_test",
    timeout = "long",
    srcs = ["networks_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    deps = [
        ":data_providers",
        ":model_configs",
        ":model_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "model_configs",
    srcs = ["model_configs.py"],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/utils:dc_constants",
    ],
)

py_library(
    name = "losses_and_metrics",
    srcs = ["losses_and_metrics.py"],
    srcs_version = "PY3",
    deps = [
        "//deepconsensus:ipython",  # buildcleaner: keep
        "//deepconsensus/utils:dc_constants",
    ],
)

py_strict_test(
    name = "losses_and_metrics_test",
    srcs = ["losses_and_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":losses_and_metrics",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "model_train_lib",
    srcs = [
        "model_train_custom_loop.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":data_providers",
        ":losses_and_metrics",
        ":model_configs",  # buildcleaner: keep
        ":model_utils",
        ":networks",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":data_providers",
        ":losses_and_metrics",
        ":networks",
        "//deepconsensus/tf_examples:tf_example_utils",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_library(
    name = "model_inference_transforms",
    srcs = ["model_inference_transforms.py"],
    srcs_version = "PY3",
    deps = [
        ":data_providers",
        ":losses_and_metrics",
        ":model_configs",
        ":model_utils",
        ":networks",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/tf_examples:tf_example_utils",
        "//deepconsensus/utils",
        "//deepconsensus/utils:colab_utils",
        "//deepconsensus/utils:dc_constants",
    ],
)

py_binary(
    name = "model_train_custom_loop",
    srcs = ["model_train_custom_loop.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":model_train_lib"],
)

py_strict_test(
    name = "model_train_custom_loop_test",
    timeout = "long",
    srcs = ["model_train_custom_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_configs",
        ":model_train_lib",
        ":model_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_binary(
    name = "model_inference_with_beam",
    srcs = ["model_inference_with_beam.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    test_lib = True,
    deps = [
        ":data_providers",
        ":losses_and_metrics",
        ":model_configs",
        ":model_inference_transforms",
        ":model_utils",
        ":networks",
        "//deepconsensus/protos:deepconsensus_py_pb2",
        "//deepconsensus/utils:dc_constants",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "model_inference_with_beam_test",
    timeout = "long",
    srcs = ["model_inference_with_beam_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_configs",
        ":model_inference_with_beam.testonly_lib",
        ":model_utils",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_strict_test(
    name = "model_inference_transforms_test",
    timeout = "long",
    srcs = ["model_inference_transforms_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_configs",
        ":model_inference_transforms",
        ":model_utils",
        "//deepconsensus/tf_examples:tf_example_utils",
        "//deepconsensus/utils:dc_constants",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_binary(
    name = "model_inference",
    srcs = ["model_inference.py"],
    data = [
        "model_configs.py",
    ],
    python_version = "PY3",
    test_lib = True,
    deps = [
        ":data_providers",
        ":losses_and_metrics",
        ":model_configs",  # buildcleaner: keep
        ":model_utils",
        ":networks",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "model_inference_test",
    timeout = "long",
    srcs = ["model_inference_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_configs",
        ":model_inference.testonly_lib",
        ":model_utils",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_strict_test(
    name = "model_utils_test",
    timeout = "long",
    srcs = ["model_utils_test.py"],
    data = [
        "//deepconsensus:testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_configs",
        ":model_utils",
        "//deepconsensus/utils:py_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
