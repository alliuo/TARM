import os
import shutil
import json
import numpy as np
from itertools import product
import pandas as pd


def Name2x2():
    """
    Name the 2x2 multipliers, the same lut has the same name
    """
    loc_list = [''.join(bits) for bits in product('hl', repeat=4)]  # hhhh, hhhl, …, llll
    df = pd.DataFrame(columns = (['mul_name', 'application', 'loc', 'lut', 'ME', 'gate_counts']))
    acc_lut = np.outer(np.arange(2**2), np.arange(2**2))
    acc_lut_str = json.dumps(acc_lut.flatten().tolist())
    name_idx = 0

    for loc in loc_list:
        tmp_df = pd.read_csv('./tmp/out_2x2/lut_2x2_' + loc + '.csv')
        for index, row in tmp_df.iterrows():
            lut_str = row['lut']
            same_lut_mul = df.loc[df.loc[:,'lut']==lut_str, 'mul_name']
            if lut_str == acc_lut_str:
                mul_name = 'mul2_acc'
            elif (same_lut_mul.empty):
                mul_name = 'mul2_' + str(name_idx)
                name_idx += 1
            else: # the same lut has the same name
                mul_name = same_lut_mul.unique()[0]
            new_row = [mul_name, row['application'], loc, lut_str, row['ME'], row['gate_counts']]
            row_len = df.shape[0]
            df.loc[row_len] = new_row
    df.to_csv('./out/lut_2x2.csv', index=False)


def GenerateRTL2x2(output_dir, mul_name, lut):
    """
    Generate the Verilog HDL accoding to the lut
    """
    mul_2x2_code = f"""
module {mul_name} (
    input  wire [1:0] a,
    input  wire [1:0] b,
    output wire [3:0] out
);

assign out = (a == 2'b00 & b == 2'b00) ? 4'b{bin(lut[0] )[2:].zfill(4)} : (
             (a == 2'b00 & b == 2'b01) ? 4'b{bin(lut[1] )[2:].zfill(4)} : (
             (a == 2'b00 & b == 2'b10) ? 4'b{bin(lut[2] )[2:].zfill(4)} : (
             (a == 2'b00 & b == 2'b11) ? 4'b{bin(lut[3] )[2:].zfill(4)} : (
             (a == 2'b01 & b == 2'b00) ? 4'b{bin(lut[4] )[2:].zfill(4)} : (
             (a == 2'b01 & b == 2'b01) ? 4'b{bin(lut[5] )[2:].zfill(4)} : (
             (a == 2'b01 & b == 2'b10) ? 4'b{bin(lut[6] )[2:].zfill(4)} : (
             (a == 2'b01 & b == 2'b11) ? 4'b{bin(lut[7] )[2:].zfill(4)} : (
             (a == 2'b10 & b == 2'b00) ? 4'b{bin(lut[8] )[2:].zfill(4)} : (
             (a == 2'b10 & b == 2'b01) ? 4'b{bin(lut[9] )[2:].zfill(4)} : (
             (a == 2'b10 & b == 2'b10) ? 4'b{bin(lut[10])[2:].zfill(4)} : (
             (a == 2'b10 & b == 2'b11) ? 4'b{bin(lut[11])[2:].zfill(4)} : (
             (a == 2'b11 & b == 2'b00) ? 4'b{bin(lut[12])[2:].zfill(4)} : (
             (a == 2'b11 & b == 2'b01) ? 4'b{bin(lut[13])[2:].zfill(4)} : (
             (a == 2'b11 & b == 2'b10) ? 4'b{bin(lut[14])[2:].zfill(4)} : (
             (a == 2'b11 & b == 2'b11) ? 4'b{bin(lut[15])[2:].zfill(4)} : 4'b0000)))))))))))))));

endmodule
"""
    out_path = os.path.join(output_dir, f'{mul_name}.v')
    with open(out_path, 'w') as f:
        f.write(mul_2x2_code)
    print(f'Generated {out_path}')
       

def GenerateTestBenchs2x2(output_dir, mul_name_list):
    """
    Generate the testbenchs for 2x2 multipliers
    """
    for mul_name in mul_name_list:
        code = f"""
`timescale 1ns / 1ps

module top_sim();
    wire [3:0] out;
    reg  [1:0] a;
    reg  [1:0] b;

    {mul_name} top (.a(a), .b(b), .out(out));

    initial
    begin
        `ifdef DUMP_VPD
                $vcdpluson();
        `endif
        a = {{$random}}%4;
        b = {{$random}}%4;
        #1000000
        `ifdef DUMP_VPD
                $vcdplusoff();
        `endif
        $finish;
    end

    always
    begin
        forever #1   begin b = {{$random}}%4;end
    end
    always
    begin
        forever #1   begin a = {{$random}}%4;end
    end

endmodule
"""
        out_path = os.path.join(output_dir, f'{mul_name}_tb.v')
        with open(out_path, 'w') as f:
            f.write(code)
        print(f'Generated {out_path}')


def NameTree():
    """
    Name the 4-bit adder trees, the same lut has the same name
    """
    loc_list = ['hh', 'hl', 'lh', 'll']
    df = pd.DataFrame(columns = (['tree_name', 'application', 'loc', 'ME', 'gate_counts', 'lut']))
    acc_fa = [(i//4)%2 + (i//2)%2 + i%2 for i in range(2**3)]  # lut of accurate full adder
    acc_ha = [(i//2)%2 + i%2 for i in range(2**2)]  # lut of accurate half adder
    acc_str = json.dumps(acc_fa*7 + acc_ha)
    name_idx = 0

    for loc in loc_list:
        tmp_df = pd.read_csv('./tmp/out_tree/lut_tree_' + loc + '.csv')
        for index, row in tmp_df.iterrows():
            lut_str = row['lut']
            same_lut_mul = df.loc[df.loc[:,'lut']==lut_str, 'tree_name']
            if lut_str == acc_str:
                tree_name = 'tree_acc'
            elif (same_lut_mul.empty):
                tree_name = 'tree_' + str(name_idx)
                name_idx += 1
            else: # the same lut has the same name
                tree_name = same_lut_mul.unique()[0]
            new_row = [tree_name, row['application'], loc, row['ME'], row['gate_counts'], lut_str]
            row_len = df.shape[0]
            df.loc[row_len] = new_row
    df.to_csv('./out/lut_tree.csv', index=False)


def GenerateRTLTree(output_dir, tree_name, fa_lut_list, rca_lut_list):
    """
    Generate the Verilog HDL accoding to the lut
    """
    tree_code = f"""
module {tree_name} (
    input  wire [3:0] pp_hh,
    input  wire [3:0] pp_hl,
    input  wire [3:0] pp_lh,
    input  wire [3:0] pp_ll,
    output wire [7:0] result
);

wire [1:0] tmp_sum3;
wire [1:0] tmp_sum2;
wire [1:0] tmp_sum1;
wire [1:0] tmp_sum0;

// Full Adder 3
assign tmp_sum3 = ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b000) ? 2'b{bin(fa_lut_list[0][0])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b001) ? 2'b{bin(fa_lut_list[0][1])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b010) ? 2'b{bin(fa_lut_list[0][2])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b011) ? 2'b{bin(fa_lut_list[0][3])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b100) ? 2'b{bin(fa_lut_list[0][4])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b101) ? 2'b{bin(fa_lut_list[0][5])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b110) ? 2'b{bin(fa_lut_list[0][6])[2:].zfill(2)} : (
                  ({{pp_hh[1], pp_hl[3], pp_lh[3]}} == 3'b111) ? 2'b{bin(fa_lut_list[0][7])[2:].zfill(2)} : 2'b00)))))));

// Full Adder 2
assign tmp_sum2 = ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b000) ? 2'b{bin(fa_lut_list[1][0])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b001) ? 2'b{bin(fa_lut_list[1][1])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b010) ? 2'b{bin(fa_lut_list[1][2])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b011) ? 2'b{bin(fa_lut_list[1][3])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b100) ? 2'b{bin(fa_lut_list[1][4])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b101) ? 2'b{bin(fa_lut_list[1][5])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b110) ? 2'b{bin(fa_lut_list[1][6])[2:].zfill(2)} : (
                  ({{pp_hh[0], pp_hl[2], pp_lh[2]}} == 3'b111) ? 2'b{bin(fa_lut_list[1][7])[2:].zfill(2)} : 2'b00)))))));

// Full Adder 1
assign tmp_sum1 = ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b000) ? 2'b{bin(fa_lut_list[2][0])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b001) ? 2'b{bin(fa_lut_list[2][1])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b010) ? 2'b{bin(fa_lut_list[2][2])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b011) ? 2'b{bin(fa_lut_list[2][3])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b100) ? 2'b{bin(fa_lut_list[2][4])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b101) ? 2'b{bin(fa_lut_list[2][5])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b110) ? 2'b{bin(fa_lut_list[2][6])[2:].zfill(2)} : (
                  ({{pp_hl[1], pp_lh[1], pp_ll[3]}} == 3'b111) ? 2'b{bin(fa_lut_list[2][7])[2:].zfill(2)} : 2'b00)))))));

// Full Adder 0
assign tmp_sum0 = ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b000) ? 2'b{bin(fa_lut_list[3][0])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b001) ? 2'b{bin(fa_lut_list[3][1])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b010) ? 2'b{bin(fa_lut_list[3][2])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b011) ? 2'b{bin(fa_lut_list[3][3])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b100) ? 2'b{bin(fa_lut_list[3][4])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b101) ? 2'b{bin(fa_lut_list[3][5])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b110) ? 2'b{bin(fa_lut_list[3][6])[2:].zfill(2)} : (
                  ({{pp_hl[0], pp_lh[0], pp_ll[2]}} == 3'b111) ? 2'b{bin(fa_lut_list[3][7])[2:].zfill(2)} : 2'b00)))))));           

wire carry0, carry1, carry2, carry3;

// RCA
assign result[2:0] = {{tmp_sum0[0], pp_ll[1:0]}};

assign {{carry0, result[3]}} = ({{tmp_sum1[0], tmp_sum0[1]}} == 2'b00) ? 2'b{bin(rca_lut_list[3][0])[2:].zfill(2)} : (
                               ({{tmp_sum1[0], tmp_sum0[1]}} == 2'b01) ? 2'b{bin(rca_lut_list[3][1])[2:].zfill(2)} : (
                               ({{tmp_sum1[0], tmp_sum0[1]}} == 2'b10) ? 2'b{bin(rca_lut_list[3][2])[2:].zfill(2)} : (
                               ({{tmp_sum1[0], tmp_sum0[1]}} == 2'b11) ? 2'b{bin(rca_lut_list[3][3])[2:].zfill(2)} : 2'b00)));  
                
assign {{carry1, result[4]}} = ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b000) ? 2'b{bin(rca_lut_list[2][0])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b001) ? 2'b{bin(rca_lut_list[2][1])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b010) ? 2'b{bin(rca_lut_list[2][2])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b011) ? 2'b{bin(rca_lut_list[2][3])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b100) ? 2'b{bin(rca_lut_list[2][4])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b101) ? 2'b{bin(rca_lut_list[2][5])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b110) ? 2'b{bin(rca_lut_list[2][6])[2:].zfill(2)} : (
                               ({{tmp_sum2[0], tmp_sum1[1], carry0}} == 3'b111) ? 2'b{bin(rca_lut_list[2][7])[2:].zfill(2)} : 2'b00)))))));

assign {{carry2, result[5]}} = ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b000) ? 2'b{bin(rca_lut_list[1][0])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b001) ? 2'b{bin(rca_lut_list[1][1])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b010) ? 2'b{bin(rca_lut_list[1][2])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b011) ? 2'b{bin(rca_lut_list[1][3])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b100) ? 2'b{bin(rca_lut_list[1][4])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b101) ? 2'b{bin(rca_lut_list[1][5])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b110) ? 2'b{bin(rca_lut_list[1][6])[2:].zfill(2)} : (
                               ({{tmp_sum3[0], tmp_sum2[1], carry1}} == 3'b111) ? 2'b{bin(rca_lut_list[1][7])[2:].zfill(2)} : 2'b00)))))));

assign {{carry3, result[6]}} = ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b000) ? 2'b{bin(rca_lut_list[0][0])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b001) ? 2'b{bin(rca_lut_list[0][1])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b010) ? 2'b{bin(rca_lut_list[0][2])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b011) ? 2'b{bin(rca_lut_list[0][3])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b100) ? 2'b{bin(rca_lut_list[0][4])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b101) ? 2'b{bin(rca_lut_list[0][5])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b110) ? 2'b{bin(rca_lut_list[0][6])[2:].zfill(2)} : (
                               ({{pp_hh[2], tmp_sum3[1], carry2}} == 3'b111) ? 2'b{bin(rca_lut_list[0][7])[2:].zfill(2)} : 2'b00)))))));

assign result[7] = pp_hh[3] | carry3;

endmodule
"""
    out_path = os.path.join(output_dir, f'{tree_name}.v')
    with open(out_path, 'w') as f:
        f.write(tree_code)
    print(f'Generated {out_path}')


def GenerateTestBenchsTree(output_dir, tree_name_list):
    """
    Generate the testbenchs for 4-bit adder trees
    """
    for tree_name in tree_name_list:
        code = f"""
`timescale 1ns / 1ps

module top_sim();
    wire [7:0] result;
    reg  [3:0] pp_hh;
    reg  [3:0] pp_hl;
    reg  [3:0] pp_lh;
    reg  [3:0] pp_ll;

    {tree_name} top (.pp_hh(pp_hh), .pp_hl(pp_hl), .pp_lh(pp_lh), .pp_ll(pp_ll), .result(result));

    initial
    begin
        `ifdef DUMP_VPD
                $vcdpluson();
        `endif
        pp_hh = {{$random}}%16;
        pp_hl = {{$random}}%16;
        pp_lh = {{$random}}%16;
        pp_ll = {{$random}}%16;
        #1000000
        `ifdef DUMP_VPD
                $vcdplusoff();
        `endif
        $finish;
    end

    always
    begin
        forever #1   begin pp_hh = {{$random}}%16;end
    end
    always
    begin
        forever #1   begin pp_hl = {{$random}}%16;end
    end
    always
    begin
        forever #1   begin pp_lh = {{$random}}%16;end
    end
    always
    begin
        forever #1   begin pp_ll = {{$random}}%16;end
    end

endmodule
"""
        out_path = os.path.join(output_dir, f'{tree_name}_tb.v')
        with open(out_path, 'w') as f:
            f.write(code)
        print(f'Generated {out_path}')


def GenerateVCSMakefile(output_dir, unit_name_list):
    """
    Generate the makefie for vcs simulation
    The simulation is used to generate the back-annotated switching activity file
    """
    with open('./config/makefile.template', 'r') as f:
        template = f.read()
    units_str = ' '.join(unit_name_list)
    code = template.format(UNITS=units_str)
    code = code.replace(r'\t', '\t')
    out_path = os.path.join(output_dir, 'makefile')
    with open(out_path, 'w') as fout:
        fout.write(code)
    print(f'Generated {out_path}')


def GenerateDCScript1(output_dir, unit_name_list):
    """
    Generate scripts for DC synthesis
    """
    with open('./config/dc_step1_template.tcl', 'r') as f:
        template = f.read()
    for unit_name in unit_name_list:
        code = template.format(UNIT_NAME=unit_name)
        out_path = os.path.join(output_dir, f'{unit_name}_step1.tcl')
        with open(out_path, 'w') as fout:
            fout.write(code)
        print(f'Generated {out_path}')


def GenerateDCScript2(output_dir, unit_name_list):
    """
    Generate scripts for DC synthesis
    """
    with open('./config/dc_step2_template.tcl', 'r') as f:
        template = f.read()
    for unit_name in unit_name_list:
        code = template.format(UNIT_NAME=unit_name)
        out_path = os.path.join(output_dir, f'{unit_name}_step2.tcl')
        with open(out_path, 'w') as fout:
            fout.write(code)
        print(f'Generated {out_path}')


def BuildSynEnv2x2():
    """
    Build the synthesis environment for 2x2 multipliers
    """
    os.makedirs('./out/', exist_ok=True)
    os.makedirs('./tmp/', exist_ok=True)
    if os.path.exists('./tmp/syn_2x2/'):
        shutil.rmtree('./tmp/syn_2x2/')
    os.mkdir('./tmp/syn_2x2')
    os.mkdir('./tmp/syn_2x2/rtl')
    os.mkdir('./tmp/syn_2x2/dc_step1')
    os.mkdir('./tmp/syn_2x2/dc_step2')
    os.mkdir('./tmp/syn_2x2/vcs')
    os.mkdir('./tmp/syn_2x2/vcs/tb')

    Name2x2()
    df_2x2 = pd.read_csv('./out/lut_2x2.csv', index_col='mul_name')
    df_2x2 = df_2x2[~df_2x2.index.duplicated(keep='first')] # remove rows with the same mul_name
    mul_name_list = df_2x2.index.unique().tolist()

    # Generate rtl
    for mul_name in mul_name_list:
        lut = json.loads(df_2x2.loc[mul_name, 'lut'])
        GenerateRTL2x2('./tmp/syn_2x2/rtl', mul_name, lut)

    # Generate syn files
    GenerateTestBenchs2x2('./tmp/syn_2x2/vcs/tb', mul_name_list)
    GenerateVCSMakefile('./tmp/syn_2x2/vcs', mul_name_list)
    GenerateDCScript1('./tmp/syn_2x2/dc_step1', mul_name_list)
    GenerateDCScript2('./tmp/syn_2x2/dc_step2', mul_name_list)


def BuildSynEnvTree():
    """
    Build the synthesis environment for 4-bit adder trees
    """
    os.makedirs('./out/', exist_ok=True)
    os.makedirs('./tmp/', exist_ok=True)
    if os.path.exists('./tmp/syn_tree/'):
        shutil.rmtree('./tmp/syn_tree/')
    os.mkdir('./tmp/syn_tree')
    os.mkdir('./tmp/syn_tree/rtl')
    os.mkdir('./tmp/syn_tree/dc_step1')
    os.mkdir('./tmp/syn_tree/dc_step2')
    os.mkdir('./tmp/syn_tree/vcs')
    os.mkdir('./tmp/syn_tree/vcs/tb')

    NameTree()
    df_tree = pd.read_csv('./out/lut_tree.csv', index_col='tree_name')
    df_tree = df_tree[~df_tree.index.duplicated(keep='first')] # remove rows with the same tree_name
    tree_name_list = df_tree.index.unique().tolist()

    # Generate rtl
    for tree_name in tree_name_list:
        lut = json.loads(df_tree.loc[tree_name, 'lut'])
        fa_lut_list = [lut[:8], lut[8:16], lut[16:24], lut[24:32]]
        rca_lut_list = [lut[32:40], lut[40:48], lut[48:56], lut[56:]]
        GenerateRTLTree('./tmp/syn_tree/rtl', tree_name, fa_lut_list, rca_lut_list)

    # Generate syn files
    GenerateTestBenchsTree('./tmp/syn_tree/vcs/tb', tree_name_list)
    GenerateVCSMakefile('./tmp/syn_tree/vcs', tree_name_list)
    GenerateDCScript1('./tmp/syn_tree/dc_step1', tree_name_list)
    GenerateDCScript2('./tmp/syn_tree/dc_step2', tree_name_list)


if __name__ == '__main__':
    BuildSynEnv2x2()
    BuildSynEnvTree()