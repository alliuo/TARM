import os
import shutil
import json
import numpy as np
from itertools import product
import pandas as pd



def GetLUT8x8(output_dir, df_2x2, df_tree, mul_name, sub_mul_list, adder_tree_list, save_flag=True):
    """
    Generate the look-up table for an 8x8 recursive multiplier
    """
    mul_lut_lists = []
    fa_lut_lists = []
    rca_lut_lists = []
    lut = np.zeros((2**8, 2**8))

    for mul in sub_mul_list:
        mul_lut_lists.append(json.loads(df_2x2.loc[mul, 'lut'])) # string to list
    
    for tree in adder_tree_list:
        lut_tree = json.loads(df_tree.loc[tree, 'lut'])
        fa_lut_lists.append([lut_tree[:8], lut_tree[8:16], lut_tree[16:24], lut_tree[24:32]])
        rca_lut_lists.append([lut_tree[32:40], lut_tree[40:48], lut_tree[48:56], lut_tree[56:]])

    if(save_flag): # save LUT to .bin file
        out_path = os.path.join(output_dir, f'{mul_name}.bin')
        with open(out_path, 'wb') as f:
            for x in range(2**8):
                for w in range(2**8):
                    lut[x, w] = SearchOutput(mul_lut_lists, fa_lut_lists, rca_lut_lists, x, w, 8)
                    f.write(int(lut[x, w]).to_bytes(2, byteorder='little', signed=False))
        print(f'Generated {out_path}')
    else:
        for x in range(2**8):
            for w in range(2**8):
                lut[x, w] = SearchOutput(mul_lut_lists, fa_lut_lists, rca_lut_lists, x, w, 8)
    return lut


def SearchOutput(mul_lut_lists, fa_lut_lists, rca_lut_lists, x, w, n):
    """
    Compute the otput of an nxn multiplication x*w
    """
    if (n == 2):
        return mul_lut_lists[0][x*4+w]
    
    half_max = int(2**(n/2))
    xh = x // half_max
    xl = x % half_max
    wh = w // half_max
    wl = w % half_max
    if(n == 4):
        pp_hh = SearchOutput([mul_lut_lists[0]], [], [], xh, wh, 2)
        pp_hl = SearchOutput([mul_lut_lists[1]], [], [], xh, wl, 2)
        pp_lh = SearchOutput([mul_lut_lists[2]], [], [], xl, wh, 2)
        pp_ll = SearchOutput([mul_lut_lists[3]], [], [], xl, wl, 2)

        tmp_sum3 = fa_lut_lists[0][0][int((pp_hh//2)%2 * 4 + (pp_hl//8)%2 * 2 + (pp_lh//8)%2)]
        tmp_sum2 = fa_lut_lists[0][1][int((pp_hh   )%2 * 4 + (pp_hl//4)%2 * 2 + (pp_lh//4)%2)]
        tmp_sum1 = fa_lut_lists[0][2][int((pp_hl//2)%2 * 4 + (pp_lh//2)%2 * 2 + (pp_ll//8)%2)]
        tmp_sum0 = fa_lut_lists[0][3][int((pp_hl   )%2 * 4 + (pp_lh   )%2 * 2 + (pp_ll//4)%2)]

        rca_tmp_sum0 = rca_lut_lists[0][3][int(tmp_sum1%2 * 2   + tmp_sum0//2)]
        rca_tmp_sum1 = rca_lut_lists[0][2][int(tmp_sum2%2 * 4   + tmp_sum1//2 * 2 + rca_tmp_sum0//2)]
        rca_tmp_sum2 = rca_lut_lists[0][1][int(tmp_sum3%2 * 4   + tmp_sum2//2 * 2 + rca_tmp_sum1//2)]
        rca_tmp_sum3 = rca_lut_lists[0][0][int((pp_hh//4)%2 * 4 + tmp_sum3//2 * 2 + rca_tmp_sum2//2)]

        result = ((pp_hh//8) | (rca_tmp_sum3//2))*128 + (rca_tmp_sum3%2)*64 + (rca_tmp_sum2%2)*32 + \
                     (rca_tmp_sum1%2)*16 + (rca_tmp_sum0%2)*8 + (tmp_sum0%2)*4 + pp_ll%4
        return result
    else:
        pp_hh = SearchOutput(mul_lut_lists[0 : 4], [fa_lut_lists[0]], [rca_lut_lists[0]], xh, wh, 4)
        pp_hl = SearchOutput(mul_lut_lists[4 : 8], [fa_lut_lists[1]], [rca_lut_lists[1]], xh, wl, 4)
        pp_lh = SearchOutput(mul_lut_lists[8 :12], [fa_lut_lists[2]], [rca_lut_lists[2]], xl, wh, 4)
        pp_ll = SearchOutput(mul_lut_lists[12:16], [fa_lut_lists[3]], [rca_lut_lists[3]], xl, wl, 4)
        return pp_hh*(2**8) + (pp_hl+pp_lh)*(2**4) + pp_ll


def GenerateRTL8x8(output_dir, mul_name, mul_2x2_list, adder_tree_list):
    mul_8x8_code = f"""
module {mul_name} (
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output wire [15:0] product
);

wire [3:0] pp_hh_hh;
wire [3:0] pp_hh_hl;
wire [3:0] pp_hh_lh;
wire [3:0] pp_hh_ll;
wire [7:0] pp_hh;

wire [3:0] pp_hl_hh;
wire [3:0] pp_hl_hl;
wire [3:0] pp_hl_lh;
wire [3:0] pp_hl_ll;
wire [7:0] pp_hl;

wire [3:0] pp_lh_hh;
wire [3:0] pp_lh_hl;
wire [3:0] pp_lh_lh;
wire [3:0] pp_lh_ll;
wire [7:0] pp_lh;

wire [3:0] pp_ll_hh;
wire [3:0] pp_ll_hl;
wire [3:0] pp_ll_lh;
wire [3:0] pp_ll_ll;
wire [7:0] pp_ll;

{mul_2x2_list[0]} mul_hh_hh (.a(a[7:6]), .b(b[7:6]), .out(pp_hh_hh));
{mul_2x2_list[1]} mul_hh_hl (.a(a[7:6]), .b(b[5:4]), .out(pp_hh_hl));
{mul_2x2_list[2]} mul_hh_lh (.a(a[5:4]), .b(b[7:6]), .out(pp_hh_lh));
{mul_2x2_list[3]} mul_hh_ll (.a(a[5:4]), .b(b[5:4]), .out(pp_hh_ll));
{adder_tree_list[0]} tree_hh (.pp_hh(pp_hh_hh), .pp_hl(pp_hh_hl), .pp_lh(pp_hh_lh), .pp_ll(pp_hh_ll), .result(pp_hh));

{mul_2x2_list[4]} mul_hl_hh (.a(a[7:6]), .b(b[3:2]), .out(pp_hl_hh));
{mul_2x2_list[5]} mul_hl_hl (.a(a[7:6]), .b(b[1:0]), .out(pp_hl_hl));
{mul_2x2_list[6]} mul_hl_lh (.a(a[5:4]), .b(b[3:2]), .out(pp_hl_lh));
{mul_2x2_list[7]} mul_hl_ll (.a(a[5:4]), .b(b[1:0]), .out(pp_hl_ll));
{adder_tree_list[1]} tree_hl (.pp_hh(pp_hl_hh), .pp_hl(pp_hl_hl), .pp_lh(pp_hl_lh), .pp_ll(pp_hl_ll), .result(pp_hl));

{mul_2x2_list[8]} mul_lh_hh (.a(a[3:2]), .b(b[7:6]), .out(pp_lh_hh));
{mul_2x2_list[9]} mul_lh_hl (.a(a[3:2]), .b(b[5:4]), .out(pp_lh_hl));
{mul_2x2_list[10]} mul_lh_lh (.a(a[1:0]), .b(b[7:6]), .out(pp_lh_lh));
{mul_2x2_list[11]} mul_lh_ll (.a(a[1:0]), .b(b[5:4]), .out(pp_lh_ll));
{adder_tree_list[2]} tree_lh (.pp_hh(pp_lh_hh), .pp_hl(pp_lh_hl), .pp_lh(pp_lh_lh), .pp_ll(pp_lh_ll), .result(pp_lh));

{mul_2x2_list[12]} mul_ll_hh (.a(a[3:2]), .b(b[3:2]), .out(pp_ll_hh));
{mul_2x2_list[13]} mul_ll_hl (.a(a[3:2]), .b(b[1:0]), .out(pp_ll_hl));
{mul_2x2_list[14]} mul_ll_lh (.a(a[1:0]), .b(b[3:2]), .out(pp_ll_lh));
{mul_2x2_list[15]} mul_ll_ll (.a(a[1:0]), .b(b[1:0]), .out(pp_ll_ll));
{adder_tree_list[3]} tree_ll (.pp_hh(pp_ll_hh), .pp_hl(pp_ll_hl), .pp_lh(pp_ll_lh), .pp_ll(pp_ll_ll), .result(pp_ll));

final_tree_acc tree (.pp_hh(pp_hh), .pp_hl(pp_hl), .pp_lh(pp_lh), .pp_ll(pp_ll), .result(product));

endmodule
"""
    out_path = os.path.join(output_dir, f'{mul_name}.v')
    with open(out_path, 'w') as f:
        f.write(mul_8x8_code)
    print(f'Generated {out_path}')


def GenerateTestBenchs8x8(output_dir, mul_name_list):
    for mul_name in mul_name_list:
        code = f"""
`timescale 1ns / 1ps

module top_sim();
    wire [15:0] product;
    reg  [7:0]  a;
    reg  [7:0]  b;

    {mul_name} top (.a(a), .b(b), .product(product));

    initial
    begin
        `ifdef DUMP_VPD
                $vcdpluson();
        `endif
        a = {{$random}}%256;
        b = {{$random}}%256;
        #1000000
        `ifdef DUMP_VPD
                $vcdplusoff();
        `endif
        $finish;
    end

    always
    begin
        forever #1   begin b = {{$random}}%256;end
    end
    always
    begin
        forever #1   begin a = {{$random}}%256;end
    end

endmodule
"""
        out_path = os.path.join(output_dir, f'{mul_name}_tb.v')
        with open(out_path, 'w') as f:
            f.write(code)


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


def GenerateDCScript1_8x8(output_dir, unit_name_list):
    """
    Generate scripts for DC synthesis
    """
    with open('./config/dc_step1_8x8_template.tcl', 'r') as f:
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


def BuildSynEnv8x8():
    """
    Build the synthesis environment for 8x8 multipliers
    """
    if os.path.exists('./tmp/syn_8x8/'):
        shutil.rmtree('./tmp/syn_8x8/')
    os.mkdir('./tmp/syn_8x8')
    os.mkdir('./tmp/syn_8x8/rtl')
    os.mkdir('./tmp/syn_8x8/dc_step1')
    os.mkdir('./tmp/syn_8x8/dc_step2')
    os.mkdir('./tmp/syn_8x8/vcs')
    os.mkdir('./tmp/syn_8x8/vcs/tb')

    if os.path.exists('./tmp/eval_8x8/'):
        shutil.rmtree('./tmp/eval_8x8/')
    os.mkdir('./tmp/eval_8x8')
    os.mkdir('./tmp/eval_8x8/lut')

    df_2x2 = pd.read_csv('./out/lut_2x2.csv', index_col='mul_name')
    df_2x2 = df_2x2[~df_2x2.index.duplicated(keep='first')] # remove rows with the same mul_name
    df_tree = pd.read_csv('./out/lut_tree.csv', index_col='tree_name')
    df_tree = df_tree[~df_tree.index.duplicated(keep='first')] # remove rows with the same tree_name
    df_8x8 = pd.read_csv('./out/mul_8x8.csv', index_col='mul_name')
    df_8x8 = df_8x8[~df_8x8.index.duplicated(keep='first')] # remove rows with the same mul_name
    mul_name_list = df_8x8.index.unique().tolist()

    # generate lut and rtl
    for mul_name, row in df_8x8.iterrows():
        structure = row['structure'].split(',')
        sub_mul_list = structure[:16]
        adder_tree_list = structure[16:]
        GetLUT8x8('./tmp/eval_8x8/lut', df_2x2, df_tree, mul_name, sub_mul_list, adder_tree_list, save_flag=True)
        GenerateRTL8x8('./tmp/syn_8x8/rtl', mul_name, sub_mul_list, adder_tree_list)

    # Generate syn files
    GenerateTestBenchs8x8('./tmp/syn_8x8/vcs/tb', mul_name_list)
    GenerateVCSMakefile('./tmp/syn_8x8/vcs', mul_name_list)
    GenerateDCScript1_8x8('./tmp/syn_8x8/dc_step1', mul_name_list)
    GenerateDCScript2('./tmp/syn_8x8/dc_step2', mul_name_list)


if __name__ == '__main__':
    BuildSynEnv8x8()