.PHONY: com sim saif clean regress

MUL := mul

OUTPUT = simv_top
ALL_DEFINE = +define+DUMP_VPD

# Code coverage command
CM = -cm line+cond+fsm+branch+tgl
CM_NAME = -cm_name ${OUTPUT}
CM_DIR = -cm_dir ./${OUTPUT}.vdb

# vpd file name
VPD_NAME = +vpdfile+${OUTPUT}.vpd

# Compile command
VCS = vcs -sverilog +v2k -timescale=1ns/1ps       \
      -debug_all                                  \
      +notimingcheck                              \
      +nospecify                                  \
      +vcs+flush+all                              \
      ${ALL_DEFINE}                               \
      ${VPD_NAME}                                 \
      -o  ${OUTPUT}                               \
      -l  compile.log                             \
      ${CM_NAME}                                  \
      ${CM_DIR}

# simulation command
SIM = ./${OUTPUT}                \
      ${CM} ${CM_NAME} ${CM_DIR} \
      ${VPD_NAME}                \
      -l ${OUTPUT}.log

# generate saif command
SAIF = vcd2saif -64 -input ${OUTPUT}.vpd -output ./saif/${MUL}.saif

# all units
units := tree_0 tree_1 tree_2 tree_3 tree_4 tree_acc tree_5 tree_6 tree_7 tree_8 tree_9 tree_10 tree_11

# compile
com:
	${VCS} /home/lao/IH28HK12T_V1p0/IH28HK_VHS_V1p0_basic/IH28HK_VHS_RVT_V1p0/verilog/IH28HK_VHS_RVT.v ../mapped/${MUL}_synthesized.v ./tb/${MUL}_tb.v;

# simulation
sim:
	${SIM}

# generate saif
saif:
	${SAIF}

clean:
	rm -rf ./csrc  *.daidir  *.log  *.vpd  *.vdb  simv*  *.key  *race.out*

# regress all units
regress:
	rm -rf ./saif; mkdir ./saif; $(foreach unit, ${units}, make clean; make com MUL=${unit}; make sim; make saif MUL=${unit};)