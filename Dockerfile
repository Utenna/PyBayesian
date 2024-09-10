# Use the official Jupyter base notebook image with Python 3.11
FROM jupyter/scipy-notebook:python-3.11

LABEL maintainer="Hu Chuan-Peng <hcp4715@hotmail.com>"

# Set environment variables to minimize Docker image size
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH="/opt/conda/bin:$PATH"

# Create a new conda environment and install packages
RUN conda install -y \
    arviz=0.17.1 \
    graphviz=0.20.1 \
    matplotlib=3.7.2 \
    numpy=2.1.1 \
    pandas=2.0.3 \
    pingouin=0.5.4 \
    pymc=5.16.2 \
    pytensor=2.25.4 \
    scipy=1.14.1 \
    seaborn=0.13.2 \
    xarray=2023.6.0 && \
    conda clean --all --yes

# Remove cache and unused packages to reduce image size
RUN rm -rf /home/jovyan/.cache && \
    conda clean --all --yes && \
    fix-permissions /home/jovyan

# Set the working directory
WORKDIR /home/jovyan/work

# Expose the default Jupyter notebook port
EXPOSE 8888

# Command to start Jupyter Notebook
CMD ["start-notebook.sh"]
