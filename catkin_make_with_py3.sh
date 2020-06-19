
catkin build

source devel/setup.bash

catkin build --cmake-args \
-DCMAKE_BUILD_TYPE=Release \
-DPYTHON_EXECUTABLE=~/miniconda3/envs/blimpRL/bin/python3 \
-DPYTHON_INCLUDE_DIR=~/miniconda3/include/python3.7m \
-DPYTHON_LIBRARY=~/miniconda3/lib/libpython3.7m.so

source devel/setup.bash