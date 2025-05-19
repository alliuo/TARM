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
    Read dara distribution file and split the 8x8 dist matrix into 2x2 dist matrices
    """
    # Read data distribution file
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
    # Split the original dist from 8x8 to 2x2
    dist = dist.reshape((2**8, 2**8))
    dist_2x2_list = SplitDist(dist, 8)
    return dist_2x2_list


def SplitDist(dist, n):
    """
    Split the nxn dist matrix into four (n/2)*(n/2) dist matrices
    """
    half_max = int(2**(n/2))
    dist_hh = np.zeros((half_max, half_max))
    dist_hl = np.zeros((half_max, half_max))
    dist_lh = np.zeros((half_max, half_max))
    dist_ll = np.zeros((half_max, half_max))
    for x in range(2**n):
        for w in range(2**n):
            xh = x // half_max
            xl = x % half_max
            wh = w // half_max
            wl = w % half_max
            dist_hh[xh, wh] += dist[x, w]
            dist_hl[xh, wl] += dist[x, w]
            dist_lh[xl, wh] += dist[x, w]
            dist_ll[xl, wl] += dist[x, w]

    if(n == 8):
        return SplitDist(dist_hh,4) + SplitDist(dist_hl,4) + SplitDist(dist_lh,4) + SplitDist(dist_ll,4)
    else:
        return [dist_hh, dist_hl, dist_lh, dist_ll]


def int_to_binary_list(a, length=2):
    """
    Convert an integer to a fixed-length binary list
    """
    binary_str = bin(a)[2:]
    binary_str = binary_str.zfill(length) # Fill high bits with 0 to meet the length
    binary_list = [int(digit) for digit in binary_str]
    return binary_list


def GetTruthTable(error_matrix):
    """
    Get the truth table based on the error matrix
    """
    acc_lut = np.outer(np.arange(2**2), np.arange(2**2))
    lut = error_matrix + acc_lut
    truth_table = []
    for x in range(0, 2**2):
        for w in range(0, 2**2):
            x_bin   = int_to_binary_list(x)
            w_bin   = int_to_binary_list(w)
            out_bin = int_to_binary_list(lut[x,w], length=4)
            row = tuple(x_bin + w_bin + out_bin)
            truth_table.append(row)
    return truth_table


def GenerateBlif(truth_table, loc):
    """
    Generate blif file based on the truth table
    """
    input_names = ['a1', 'a0', 'b1', 'b0']
    output_names = ['y3', 'y2', 'y1', 'y0']
    with open(f"./tmp/blif_2x2/{loc}_circuit.blif", 'w') as f:
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


def RunABC(abc_path, loc):
    """
    Run ABC for logic optimization, the ABC script referred to BLASYS (https://github.com/scale-lab/BLASYS)
    """
    script = f"read ./tmp/blif_2x2/{loc}_circuit.blif; strash; ifraig; dc2; fraig; rewrite; refactor; resub; rewrite; refactor; resub; rewrite; rewrite -z; rewrite -z; rewrite -z; balance; refactor -z; refactor; balance; write ./tmp/blif_2x2/{loc}_optimized_circuit.blif"
    process = subprocess.Popen([abc_path, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"ABC execution failed: {stderr.decode()}")
    return stdout.decode()


def ParseBlif():
    """
    Parse BLIF files to count logic gate counts and types
    """
    with open(f"./tmp/blif_2x2/{loc}_optimized_circuit.blif", 'r') as file:
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


def MinimizeLogic(abc_path, error_matrix, loc):
    truth_table = GetTruthTable(error_matrix)
    GenerateBlif(truth_table, loc)
    RunABC(abc_path, loc)
    gate_count, logic_gates = ParseBlif()
    return gate_count


def NSGA2x2(abc_path, loc, dist, sub_ME_constraint, pop_size, ngen, cxpb, mutpb):
    """
    Find Pareto Front through NSGA-II Algorithm
    """
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    acc_lut_flatten = np.outer(np.arange(2**2), np.arange(2**2)).flatten()

    def generate_individual():
        while True:
            individual = [random.randint(-i, 0) for i in acc_lut_flatten]
            error_matrix = np.array(individual).reshape(2**2, 2**2)
            ME = np.sum(error_matrix * dist)
            if sub_ME_constraint[0] <= ME <= sub_ME_constraint[1]:
                print(individual)
                return individual

    def evaluate(individual):
        error_matrix = np.array(individual).reshape(2**2, 2**2)
        ME = np.sum(error_matrix * dist)
        if not (sub_ME_constraint[0] <= ME <= sub_ME_constraint[1]):
            ME = -5000000
            gate_counts = 5000000
        else:
            gate_counts = MinimizeLogic(abc_path, error_matrix, loc)
        return np.abs(ME), gate_counts

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selNSGA2)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low = (-1*acc_lut_flatten).tolist(), up = [0]*2**4, indpb=0.3)

    population = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=2*pop_size, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=None, halloffame=None, verbose=True)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    return pareto_front


def Optimize2x2(loc, div, loc_index, pop_size=200, ngen=500, cxpb=0.8, mutpb=0.2):
    # Create temporary folders to store intermediate data
    for d in ('./tmp', './tmp/blif_2x2', './tmp/out_2x2'):
        os.makedirs(d, exist_ok=True)

    config_inst = Config()

    # Get the data distribution matrix corresponding to this 2x2 multiplier
    dist = GetSubDist(config_inst.dist_file)[loc_index]

    # Get the constraint for sub-ME (the ME of 2x2 multiplier)
    sub_ME_constraint = [config_inst.ME_constraint[0]/div, config_inst.ME_constraint[1]/div]

    # Optimize 2x2 multipliers and record the results
    df = pd.DataFrame(columns = (['mul_name', 'application', 'loc', 'lut', 'ME', 'gate_counts']))
    name_idx = 0
    print('------------------',loc,'-------------------')
    if loc.startswith('hh'): 
        unique_pareto_front = [[0]*16] # using exact mul for the MSBs
    else:
        T1 = time.perf_counter()
        pareto_front = NSGA2x2(config_inst.abc, loc, dist, sub_ME_constraint, pop_size, ngen, cxpb, mutpb)
        T2 = time.perf_counter()
        run_time = T2-T1
        print(f"Total Time spent: {run_time:.4f} seconds")
        print('----------------------------------------')

        unique_pareto_front = set(tuple(sublist) for sublist in pareto_front)
        unique_pareto_front = [list(sublist) for sublist in unique_pareto_front]
    print(unique_pareto_front)

    for error_matrix in unique_pareto_front:
        error_matrix = np.array(error_matrix).reshape(2**2, 2**2)
        ME = np.sum(error_matrix * dist)
        if not (sub_ME_constraint[0] <= ME <= sub_ME_constraint[1]): # Violations may occur due to mutations
            continue
        gate_counts = MinimizeLogic(config_inst.abc, error_matrix, loc)
        lut = error_matrix + np.outer(np.arange(2**2), np.arange(2**2))
        lut_str = json.dumps(lut.flatten().tolist()) # list to string
        mul_name = 'tmp_mul2_' + str(name_idx)
        name_idx += 1
        new_row = [mul_name, config_inst.application, loc, lut_str, ME, gate_counts]
        row_len = df.shape[0]
        df.loc[row_len] = new_row  # Add new_row to df
        print(new_row)

    df.to_csv('./tmp/out_2x2/lut_2x2_' + loc + '.csv', index=False)


if __name__ == "__main__":
    loc_dict = {'hhhh':[2**12, 0], 'hhhl':[2**10, 1], 'hhlh':[2**10, 2], 'hhll':[2**8, 3], 
                'hlhh':[2**8, 4],  'hlhl':[2**6, 5],  'hllh':[2**6, 6],  'hlll':[2**4, 7], 
                'lhhh':[2**8, 8],  'lhhl':[2**6, 9],  'lhlh':[2**6, 10], 'lhll':[2**4, 11], 
                'llhh':[2**4, 12], 'llhl':[2**2, 13], 'lllh':[2**2, 14], 'llll':[2**0, 15]} # loc:[ME_div, index]

    # Input Parameters
    parser = argparse.ArgumentParser(description="Run 2x2 multiplier optimization within an 8x8 multiplier.")
    parser.add_argument(
        '--loc',
        type=str,
        choices=loc_dict.keys(),
        default='llll',
        help=(
            "Location of the 2x2 multiplier in the 8x8 structure. "
            "One of: " + ', '.join(loc_dict.keys()) +
            ". Example: 'hhhh' represents the 2x2 multiplier is used to implement a_HHHH * b_HHHH"
        )
    )
    args = parser.parse_args()
    loc = args.loc
    div, loc_index = loc_dict[loc]

    Optimize2x2(loc, div, loc_index, pop_size=200, ngen=500, cxpb=0.8, mutpb=0.2)
