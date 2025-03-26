# On Atos
module load python3/3.11.10-01
module load nvidia/24.11
python3 -m venv $PERM/venvs/aifs-fw-bw

source $PERM/venvs/aifs-fw-bw/bin/activate
git clone -b hackathon2024 git@github.com:ecmwf/anemoi-core.git
cat "source $PERM/venvs/aifs-fw-bw/bin/activate" > env.sh
cat "module load nvidia/24.11" >> env.sh #for nsys

#install the model code, this is where the mappers and processor is implemented
cd anemoi-core/models
pip install -e .
cd -
#Install the training code. we need this in order to build models from config
cd anemoi-core/training
pip install -e .
cd -

/perm/naco/scripts/get-flash-attn #this installs the 'flash-attn' library, which is recomended when using the transformer processor

ln -s anemoi-core/training/src/anemoi/training/config . # aifs-fw-bw looks for configs in ./config

cp -r /perm/naco/aifs/graphs . # aifs-fw-bw currently needs premade graphs

