conda create -n ulab python==3.6 -y
source activate ulab
conda install -c anaconda mkl -y
conda install -c conda-forge jupyterlab -y
conda install matplotlib -y
conda install pytorch torchvision -c pytorch -y
pip install -U pillow 
