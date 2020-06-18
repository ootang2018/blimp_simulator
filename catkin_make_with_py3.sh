
catkin build

source devel/setup.bash

catkin build --cmake-args \
-DCMAKE_BUILD_TYPE=Release \
-DPYTHON_EXECUTABLE=/home/yliu_local/miniconda3/envs/blimpRL/bin/python3 \
-DPYTHON_INCLUDE_DIR=/home/yliu_local/miniconda3/include/python3.7m \
-DPYTHON_LIBRARY=/home/yliu_local/miniconda3/lib/libpython3.7m.so

source devel/setup.bash