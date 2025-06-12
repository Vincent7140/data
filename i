#!/bin/bash

# note: launch this script via ". ./env.sh"

# Crée l'environnement (Python 3.8)
conda create -y -n SatelliteSfM python=3.8
conda activate SatelliteSfM

# Pip packages utiles
pip install numpy matplotlib opencv-python pyexr open3d tqdm icecream imageio imageio-ffmpeg
pip install utm pyproj pymap3d
pip install trimesh pyquaternion srtm4

# GDAL via conda-forge (géodonnées)
conda install -y -c conda-forge gdal

# Lib TIFF (image)
conda install -y -c anaconda libtiff

# Compiler (GCC 7, car requis par projet)
conda install -y -c conda-forge gcc_linux-64=7 gxx_linux-64=7

# Outils de build
conda install -y -c conda-forge cmake make ninja pkg-config unzip git wget

# Dépendances C++ requises mais non gérées par le script Python
conda install -y -c conda-forge boost qt-main

# Compiler avec GCC Conda
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

# Inclure les chemins d’en-têtes et de librairies
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
export QT_QPA_PLATFORM_PLUGIN_PATH="$CONDA_PREFIX/plugins/platforms"

echo ""
echo "✅ Environnement SatelliteSfM prêt"
echo "➡️ Tu peux maintenant lancer :"
echo ""
echo "    bash ./preprocess_sfm/install_colmapforvissat.sh"




conda install -c conda-forge \
  cmake=3.25 \
  boost=1.78 \
  eigen=3.3.5 \
  ceres-solver=1.14.0 \
  glog=0.3.5 \
  gflags=2.2.1 \
  glew=2.1.0 \
  freeimage=3.18.0 \
  qt=5.15 \
  libglu \
  libtiff \
  suitesparse
