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
units := mul8_0 mul8_1 mul8_2 mul8_3 mul8_4 mul8_5 mul8_6 mul8_7 mul8_8 mul8_9 mul8_10 mul8_11 mul8_12 mul8_13 mul8_14 mul8_15 mul8_16 mul8_17 mul8_18 mul8_19 mul8_20 mul8_21 mul8_22 mul8_23 mul8_24 mul8_25 mul8_26 mul8_27 mul8_28 mul8_29 mul8_30 mul8_31 mul8_32 mul8_33 mul8_34 mul8_35 mul8_36 mul8_37 mul8_38 mul8_39 mul8_40 mul8_41 mul8_42 mul8_43 mul8_44 mul8_45 mul8_46 mul8_47 mul8_48 mul8_49 mul8_50 mul8_51 mul8_52 mul8_53 mul8_54 mul8_55 mul8_56 mul8_57 mul8_58 mul8_59 mul8_60 mul8_61 mul8_62 mul8_63 mul8_64 mul8_65 mul8_66 mul8_67 mul8_68 mul8_69 mul8_70 mul8_71 mul8_72 mul8_73 mul8_74 mul8_75 mul8_76 mul8_77 mul8_78 mul8_79 mul8_80 mul8_81 mul8_82 mul8_83 mul8_84 mul8_85 mul8_86 mul8_87 mul8_88 mul8_89 mul8_90 mul8_91 mul8_92 mul8_93 mul8_94 mul8_95 mul8_96 mul8_97 mul8_98 mul8_99 mul8_100 mul8_101 mul8_102 mul8_103 mul8_104 mul8_105 mul8_106 mul8_107 mul8_108 mul8_109 mul8_110 mul8_111 mul8_112 mul8_113 mul8_114 mul8_115 mul8_116 mul8_117 mul8_118 mul8_119 mul8_120 mul8_121 mul8_122 mul8_123 mul8_124 mul8_125 mul8_126 mul8_127 mul8_128 mul8_129 mul8_130 mul8_131 mul8_132 mul8_133 mul8_134 mul8_135 mul8_136 mul8_137 mul8_138 mul8_139 mul8_140 mul8_141 mul8_142 mul8_143 mul8_144 mul8_145 mul8_146 mul8_147 mul8_148 mul8_149 mul8_150 mul8_151

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