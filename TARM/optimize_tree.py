import os
import argparse
import json
import time
import random
import subprocess
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from config import Config



def GetSubDist(dist_file):
    """
    Read dara distribution file and split the 8x8 dist matrix into four 4x4 dist matrices
    """
    # Read dist file
    dist = np.zeros(2**16)
    index = 0
    with open(dist_file) as f:
        for line in f.readlines():
            if line != '\n':
                dist[index] = float(line.strip())
                index += 1
        if(index != 2**16):
            print('Dist File Error')
            return
    # Split the original dist from 8x8 to 4x4
    dist = dist.reshape((2**8, 2**8))
    half_max = int(2**(8/2))
    dist_hh = np.zeros((half_max, half_max))
    dist_hl = np.zeros((half_max, half_max))
    dist_lh = np.zeros((half_max, half_max))
    dist_ll = np.zeros((half_max, half_max))
    for x in range(2**8):
        for w in range(2**8):
            xh = x // half_max
            xl = x % half_max
            wh = w // half_max
            wl = w % half_max
            dist_hh[xh, wh] += dist[x, w]
            dist_hl[xh, wl] += dist[x, w]
            dist_lh[xl, wh] += dist[x, w]
            dist_ll[xl, wl] += dist[x, w]
    dist_4x4_list = [dist_hh, dist_hl, dist_lh, dist_ll]
    return dist_4x4_list


def GetTreeME(dist, mul_lut_list, fa_lut_list, rca_lut_list):
    """
    Evaluate the ME of a 4-bit adder tree
    """
    ME_tree = 0
    half_max = int(2**2)

    for x in range(2**4):
        for w in range(2**4):
            xh = x // half_max
            xl = x % half_max
            wh = w // half_max
            wl = w % half_max
            pp_hh = mul_lut_list[0][xh, wh]
            pp_hl = mul_lut_list[1][xh, wl]
            pp_lh = mul_lut_list[2][xl, wh]
            pp_ll = mul_lut_list[3][xl, wl]

            tmp_sum3 = fa_lut_list[0][int((pp_hh//2)%2 * 4 + (pp_hl//8)%2 * 2 + (pp_lh//8)%2)]
            tmp_sum2 = fa_lut_list[1][int((pp_hh   )%2 * 4 + (pp_hl//4)%2 * 2 + (pp_lh//4)%2)]
            tmp_sum1 = fa_lut_list[2][int((pp_hl//2)%2 * 4 + (pp_lh//2)%2 * 2 + (pp_ll//8)%2)]
            tmp_sum0 = fa_lut_list[3][int((pp_hl   )%2 * 4 + (pp_lh   )%2 * 2 + (pp_ll//4)%2)]

            rca_tmp_sum0 = rca_lut_list[3][int(tmp_sum1%2 * 2   + tmp_sum0//2)]
            rca_tmp_sum1 = rca_lut_list[2][int(tmp_sum2%2 * 4   + tmp_sum1//2 * 2 + rca_tmp_sum0//2)]
            rca_tmp_sum2 = rca_lut_list[1][int(tmp_sum3%2 * 4   + tmp_sum2//2 * 2 + rca_tmp_sum1//2)]
            rca_tmp_sum3 = rca_lut_list[0][int((pp_hh//4)%2 * 4 + tmp_sum3//2 * 2 + rca_tmp_sum2//2)]

            result = ((pp_hh//8) | (rca_tmp_sum3//2))*128 + (rca_tmp_sum3%2)*64 + (rca_tmp_sum2%2)*32 + \
                     (rca_tmp_sum1%2)*16 + (rca_tmp_sum0%2)*8 + (tmp_sum0%2)*4 + pp_ll%4
            
            error = result - (pp_hh*(2**4) + (pp_hl+pp_lh)*(2**2) + pp_ll)
            ME_tree += dist[x,w] * error

    return ME_tree


def int_to_binary_list(a, length):
    """
    Convert an integer to a fixed-length binary list
    """
    binary_str = bin(a)[2:]
    binary_str = binary_str.zfill(length) # Fill high bits with 0 to meet the length
    binary_list = [int(digit) for digit in binary_str]
    return binary_list


def GetTruthTableFA(lut):
    """
    Get the truth table based on the LUT for a full adder
    """
    truth_table = []
    for input in range(0, 2**3):
        in_bin  = int_to_binary_list(input, length=3)
        out_bin = int_to_binary_list(lut[input], length=2)
        row = tuple(in_bin + out_bin)
        truth_table.append(row)
    return truth_table


def GetTruthTableHA(lut):
    """
    Get the truth table based on the LUT for a half adder
    """
    truth_table = []
    for input in range(0, 2**2):
        in_bin  = int_to_binary_list(input, length=2)
        out_bin = int_to_binary_list(lut[input], length=2)
        row = tuple(in_bin + out_bin)
        truth_table.append(row)
    return truth_table


def GenerateBlif(truth_table, input_names, output_names, file_name):
    """
    Generate blif file based on the truth table
    """
    with open(file_name, 'w') as f:
        f.write(".model top\n")
        f.write(".inputs " + " ".join(input_names) + "\n")
        f.write(".outputs " + " ".join(output_names) + "\n")
        for output_index, output_name in enumerate(output_names):
            if all(row[len(input_names) + output_index] == 0 for row in truth_table): # If the output is always 0
                f.write(".names " + output_name + "\n")
                f.write("0\n")
            else:
                f.write(".names " + " ".join(input_names) + " " + output_name + "\n")
                for row in truth_table:
                    input_values = "".join(map(str, row[:len(input_names)]))
                    output_value = str(row[len(input_names) + output_index])
                    if output_value == '1':
                        f.write(input_values + " " + output_value + "\n")
        f.write(".end\n")


def RunABC(abc_path, input_files, output_files):
    """
    Run ABC for logic optimization, the ABC script referred to BLASYS (https://github.com/scale-lab/BLASYS)
    """
    assert len(input_files) == len(output_files), "Input and output file lists must be the same length."
    script = ""
    for input_file, output_file in zip(input_files, output_files):
        script += f"read {input_file}; strash; ifraig; dc2; fraig; rewrite; refactor; resub; rewrite; refactor; resub; rewrite; rewrite -z; rewrite -z; rewrite -z; balance; refactor -z; refactor; balance; write {output_file}; "
    process = subprocess.Popen([abc_path, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"ABC execution failed: {stderr.decode()}")
    return stdout.decode()


def ParseBlif(file_name):
    """
    Parse BLIF files to count logic gate counts and types
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

    logic_gates = {}
    gate_count = 0

    for line in lines:
        if line.startswith('.names'):
            inputs = line.split()[1:-1]
            num_inputs = len(inputs)
            if num_inputs == 0:
                gate_type = "Wire"
            elif num_inputs == 1:
                gate_type = "NOT/Wire"
            elif num_inputs == 2:
                gate_type = "AND/OR"
                gate_count += 1 # Only two-input logic gates will be counted
            else:
                gate_type = f"Complex ({num_inputs}-input)"
                print(gate_type)
            if gate_type not in logic_gates:
                logic_gates[gate_type] = 0
            logic_gates[gate_type] += 1

    return gate_count, logic_gates


def MinimizeLogic(abc_path, loc, fa_lut_list, rca_lut_list, enable_print=False):
    input_files = [f"./tmp/blif_tree/{loc}_fa{i}_circuit.blif" for i in range(4)] + \
                  [f"./tmp/blif_tree/{loc}_rca_fa{i}_circuit.blif" for i in range(3)] + \
                  [f"./tmp/blif_tree/{loc}_rca_ha_circuit.blif"]
    
    output_files = [f"./tmp/blif_tree/{loc}_fa{i}_opt_circuit.blif" for i in range(4)] + \
                   [f"./tmp/blif_tree/{loc}_rca_fa{i}_opt_circuit.blif" for i in range(3)] + \
                   [f"./tmp/blif_tree/{loc}_rca_ha_opt_circuit.blif"]
    
    T1 = time.perf_counter()
    idx = 0
    for fa_lut in fa_lut_list + rca_lut_list[:3]:
        truth_table_fa = GetTruthTableFA(fa_lut)
        GenerateBlif(truth_table_fa, ['cin', 'a', 'b'], ['carry', 'sum'], input_files[idx])
        idx += 1
    truth_table_ha = GetTruthTableHA(rca_lut_list[3])
    GenerateBlif(truth_table_ha, ['a', 'b'], ['carry', 'sum'], input_files[idx])

    RunABC(abc_path, input_files, output_files)

    total_gate_count = 0
    for output_file in output_files:
        gate_count, logic_gates = ParseBlif(output_file)
        total_gate_count += gate_count
        if(enable_print):
            print(gate_count)
    T2 = time.perf_counter()
    run_time = T2-T1
    if(enable_print):
        print(f"Minimize Logic time spent: {run_time:.4f} seconds")
        print('----------------------------------------')
    return total_gate_count


def NSGATree(abc_path, loc, dist, mul_lut_list, sub_ME_constraint, pop_size, ngen, cxpb, mutpb):
    """
    Find Pareto Front through NSGA-II Algorithm
    """
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    acc_fa = [(i//4)%2 + (i//2)%2 + i%2 for i in range(2**3)]
    acc_ha = [(i//2)%2 + i%2 for i in range(2**2)]
    def generate_individual():
        while True:
            rand_num_fa = random.randint(0, 30)
            rand_num_rca = random.randint(0, 30)

            fa_lut_list = [[random.randint(0, i) for i in acc_fa] for j in range(4)]
            if (rand_num_fa in [0, 1]):
                fa_lut_list = [[0, 1, 1] + [random.randint(0,2)] + [1, 2, 2] + [random.randint(0,3)] for j in range(4)]
            elif (rand_num_fa == 2):
                fa_lut_list = [acc_fa[:-1] + [random.randint(0,3)] for j in range(4)]
                
            rca_lut_list = rca_lut_list = [acc_fa]*3 + [acc_ha]
            if (rand_num_rca == 0):
                rca_lut_list[0] = [0, 1, 1] + [random.randint(0,2)] + [1, 2, 2] + [random.randint(0,3)]
                rca_lut_list[1] = acc_fa[:-1] + [random.randint(0,3)]
                rca_lut_list[2] = [0, 1, 1] + [random.randint(0,2)] + [1, 2, 2, 3]
            elif (rand_num_rca in [1, 2, 3]):
                rca_lut_list[:3] = [[0, 1, 1] + [random.randint(0,2)] + [1, 2, 2] + [random.randint(0,3)] for j in range(3)]
            elif (rand_num_rca in [4, 5, 6]):
                rca_lut_list[:3] = [[0, 1, 1] + [random.randint(0,2)] + [1, 2, 2] + [random.randint(0,3)] for j in range(3)]
                rca_lut_list[3] = [0, 1, 1, 1]
            elif (rand_num_rca in [7, 8, 9]):
                rca_lut_list[:3] = [[0] + [random.randint(0,1)] + [1] + [random.randint(0,2)] + \
                                    [1] + [random.randint(0,2)] + [2] + [random.randint(0,3)] for j in range(3)]
            elif (rand_num_rca == 10):
                rca_lut_list[3] = [random.randint(0, i) for i in acc_ha]
           
            ME = GetTreeME(dist, mul_lut_list, fa_lut_list, rca_lut_list)
            if sub_ME_constraint[0] <= ME <= sub_ME_constraint[1]:
                tmp_individual = fa_lut_list + rca_lut_list
                print(tmp_individual)
                print(ME)
                individual = [item for sublist in tmp_individual for item in sublist]  # to one dimension
                return individual

    def evaluate(individual):
        fa_lut_list = [individual[:8], individual[8:16], individual[16:24], individual[24:32]]
        rca_lut_list = [individual[32:40], individual[40:48], individual[48:56], individual[56:]]
        ME = GetTreeME(dist, mul_lut_list, fa_lut_list, rca_lut_list)
        if not (sub_ME_constraint[0] <= ME <= sub_ME_constraint[1]):
            ME = -5000000
            gate_counts = 5000000
        else:
            gate_counts = MinimizeLogic(abc_path, loc, fa_lut_list, rca_lut_list)
        return np.abs(ME), gate_counts

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selNSGA2)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0]*(2**3*7 + 2**2), up=acc_fa*7+acc_ha, indpb=0.3)

    population = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=2*pop_size, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=None, halloffame=None, verbose=True)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    return pareto_front


def OptimizeTree(loc, div, loc_index, pop_size=200, ngen=500, cxpb=0.8, mutpb=0.2):
    # Create temporary folders to store intermediate data
    for d in ('./tmp', './tmp/blif_tree', './tmp/out_tree'):
        os.makedirs(d, exist_ok=True)
    
    config_inst = Config()
    
    # Get the data distribution matrix corresponding to this 4-bit adder tree
    dist = GetSubDist(config_inst.dist_file)[loc_index]

    # Get the constraint for sub-ME (the ME of 4-bit adder tree)
    sub_ME_constraint = [config_inst.ME_constraint[0]/div, config_inst.ME_constraint[1]/div]

    # Assuming the preceding 2x2 multipliers are exact
    mul_luc_acc = np.outer(np.arange(2**2), np.arange(2**2))
    mul_lut_list = [mul_luc_acc] * 4
    
    # Optimize 4-bit adder trees and record the results
    df = pd.DataFrame(columns = (['tree_name', 'application', 'loc', 'ME', 'gate_counts', 'lut']))
    name_idx = 0
    print('------------------',loc,'-------------------')
    T1 = time.perf_counter()
    pareto_front = NSGATree(config_inst.abc, loc, dist, mul_lut_list, sub_ME_constraint, pop_size, ngen, cxpb, mutpb)
    T2 = time.perf_counter()
    run_time = T2-T1
    print(f"Total Time spent: {run_time:.4f} seconds")
    print('----------------------------------------')

    acc_fa = [(i//4)%2 + (i//2)%2 + i%2 for i in range(2**3)] # lut of accurate full adder
    acc_ha = [(i//2)%2 + i%2 for i in range(2**2)]  # lut of accurate half adder
    pareto_front += [acc_fa*7 + acc_ha] # Add the accurate adder tree into the pareto front

    unique_pareto_front = set(tuple(sublist) for sublist in pareto_front)
    unique_pareto_front = [list(sublist) for sublist in unique_pareto_front]
    print(unique_pareto_front)

    for lut in unique_pareto_front:
        fa_lut_list = [lut[:8], lut[8:16], lut[16:24], lut[24:32]]
        rca_lut_list = [lut[32:40], lut[40:48], lut[48:56], lut[56:]]
        ME = GetTreeME(dist, mul_lut_list, fa_lut_list, rca_lut_list)
        if not (sub_ME_constraint[0] <= ME <= sub_ME_constraint[1]): # Violations may occur due to mutations
            continue
        gate_counts = MinimizeLogic(config_inst.abc, loc, fa_lut_list, rca_lut_list, enable_print=True)
        lut_str = json.dumps(lut) # list to string
        tree_name = 'tmp_tree_' + str(name_idx)
        name_idx += 1
        new_row = [tree_name, config_inst.application, loc, ME, gate_counts, lut_str]
        row_len = df.shape[0]
        df.loc[row_len] = new_row  # Add new_row to df
        print(new_row)

    df.to_csv('./tmp/out_tree/lut_tree_' + loc + '.csv', index=False)


if __name__ == "__main__":
    loc_dict = {'hh':[2**8, 0], 'hl':[2**4, 1], 'lh':[2**4, 2], 'll':[2**0, 3]} # loc:[div, index]

    # Input Parameters
    parser = argparse.ArgumentParser(description="Run 4-bit adder tree optimization within an 8x8 multiplier.")
    parser.add_argument(
        '--loc',
        type=str,
        choices=loc_dict.keys(),
        default='ll',
        help=(
            "Location of the 4-bit adder tree in the 8x8 structure. "
            "One of: " + ', '.join(loc_dict.keys()) +
            ". Example: 'hh' represents the 4-bit adder tree used in a 4x4 multiplier implementing a_HH * b_HH"
        )
    )
    args = parser.parse_args()
    loc = args.loc
    div, loc_index = loc_dict[loc]

    OptimizeTree(loc, div, loc_index, pop_size=200, ngen=500, cxpb=0.8, mutpb=0.2)
