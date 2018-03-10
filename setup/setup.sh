conda create env -n ulab python==3.6 -y
source activate ulab
conda install -c intel mkl -y
conda install matplotlib -y
conda install pytorch torchvision -c pytorch -y
pip install -U pillow 