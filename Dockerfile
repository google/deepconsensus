# Build with:
#   sudo docker build -t deepconsensus .
# For GPU:
#   sudo docker build --build-arg build_gpu=true --build-arg FROM_IMAGE=nvidia/cuda:11.3.0-cudnn8-runtime -t deepconsensus_gpu .



ARG FROM_IMAGE=continuumio/miniconda3

FROM continuumio/miniconda3 as conda_setup
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge
RUN conda create -n bio \
                    python=3.8 \
                    pbcore \
                    pbbam \
                    pbccs \
                    pbmm2 \
                    parallel \
                    jq \
                    gcc \
                    memory_profiler \
                    pycocotools \
                    bioconda::seqtk \
                    bioconda::bedtools \
                    bioconda::minimap2 \
                    bioconda::extracthifi \
                    bioconda::zmwfilter \
                    bioconda::pysam \
                    bioconda::samtools=1.10 \
    && conda clean -a
RUN wget https://github.com/PacificBiosciences/align-clr-to-ccs/releases/download/0.2.0/actc && \
    chmod +x actc && \
    mv actc /opt/conda/bin/

FROM ${FROM_IMAGE} as builder
COPY --from=conda_setup /opt/conda /opt/conda

ENV PATH=/opt/conda/envs/bio/bin:/opt/conda/bin:"${PATH}"
ENV LD_LIBRARY_PATH=/opt/conda/envs/bio/lib:/opt/mytools/lib/x86_64-linux-gnu:"${LD_LIBRARY_PATH}"

COPY . /opt/deepconsensus
WORKDIR /opt/deepconsensus
ARG build_gpu
RUN if [ "${_TAG_NAME}" = "*gpu" ] || [ "${build_gpu}" = "true" ]; then \
        echo "Installing deepconsensus[gpu] version"; \
        pip install .[gpu]; \
    else \
        echo "Installing deepconsensus[cpu] version"; \
        pip install .[cpu]; \
    fi

CMD ["deepconsensus"]

