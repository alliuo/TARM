#!/bin/csh -f

cd /home/lao/lut_gen_public/TARM/tmp/syn_8x8/vcs

#This ENV is used to avoid overriding current script in next vcselab run 
setenv SNPS_VCSELAB_SCRIPT_NO_OVERRIDE  1

/home/opt/Synopsys/vcs_2022.06/T-2022.06/linux/bin/vcselab $* \
    -o \
    simv_top \
    -nobanner \

cd -

