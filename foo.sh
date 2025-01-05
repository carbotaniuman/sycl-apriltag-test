#!/bin/bash

while true; do
    RUSTICL_ENABLE=llvmpipe RUSTICL_DEVICE_TYPE=gpu ./sycl_apriltag_test a

    if [ $? -ne 0 ]; then
        echo "Segmentation fault occurred. Stopping."
        break
    fi

    echo "No segfault, running again."
done
