#!/bin/bash
rm -f yolov5_demo.arm
rm -f yolov5_demo.pcie
make -f Makefile.pcie && make -f Makefile.arm
