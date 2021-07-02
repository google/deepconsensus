package(
    default_visibility = [
        "//visibility:public",
    ],
)

# ------------------------------------------------------------------------------
# deepconsensus proto
# ------------------------------------------------------------------------------

py_proto_library(
    name = "deepconsensus_py_pb2",
    api_version = 2,
    deps = [":deepconsensus_proto"],
)

proto_library(
    name = "deepconsensus_proto",
    srcs = ["deepconsensus.proto"],
    cc_api_version = 2,
)

pyclif_proto_library(
    name = "deepconsensus_pyclif",
    proto_lib = ":deepconsensus_proto",
)
