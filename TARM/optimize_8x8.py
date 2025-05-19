import json
import time
import random
import numpy as np
import pandas as pd
from itertools import product
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


def GetErrorMatrix4bit(tree_name, fa_lut_list, rca_lut_list):
    """
    Generate the error matrix for a 4-bit adder tree 
    """
    print(tree_name)
    error_matrix = np.zeros((2**4, 2**4, 2**4, 2**4))

    for pp_hh in range(2**4):
        for pp_hl in range(2**4):
            for pp_lh in range(2**4):
                for pp_ll in range(2**4):
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
                    error_matrix[pp_hh, pp_hl, pp_lh, pp_ll] = result - (pp_hh*(2**4) + (pp_hl+pp_lh)*(2**2) + pp_ll)
    return error_matrix


def GetME4x4(df_2x2_list, idx_list, dist, tree_error_matrix):
    """
    Calculate the ME for an 4x4 recursive multiplier
    """
    mul_lut_list = []
    mul_ME_list = []
    for i in range (4):
        lut = json.loads(df_2x2_list[i].loc[idx_list[i], 'lut'])
        lut = np.array(lut).reshape(2**2, 2**2)
        ME = df_2x2_list[i].loc[idx_list[i], 'ME']
        mul_lut_list.append(lut)
        mul_ME_list.append(ME)

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
            error = tree_error_matrix[pp_hh, pp_hl, pp_lh, pp_ll]
            ME_tree += dist[x,w] * error
    return ME_tree + mul_ME_list[0]*(2**4) + (mul_ME_list[1]+mul_ME_list[2])*(2**2) + mul_ME_list[3]


def GetME(df_2x2_list, idx_list, dist_list, tree_error_matrices_list):
    """
    Calculate the ME for an 8x8 recursive multiplier
    """
    ME_hh = GetME4x4(df_2x2_list[  : 4], idx_list[  : 4], dist_list[0], tree_error_matrices_list[0][idx_list[16]])
    ME_hl = GetME4x4(df_2x2_list[4 : 8], idx_list[4 : 8], dist_list[1], tree_error_matrices_list[1][idx_list[17]])
    ME_lh = GetME4x4(df_2x2_list[8 :12], idx_list[8 :12], dist_list[2], tree_error_matrices_list[2][idx_list[18]])
    ME_ll = GetME4x4(df_2x2_list[12:16], idx_list[12:16], dist_list[3], tree_error_matrices_list[3][idx_list[19]])
    return ME_hh*(2**8) + (ME_hl+ME_lh)*(2**4) + ME_ll


def NSGA8x8(df_2x2_list, df_tree_list, tree_error_matrices_list, mul_dist_list, ME_constraint, pop_size, ngen, cxpb, mutpb):
    """
    Find Pareto-optimal designs through NSGA-II Algorithm
    """
    max_2x2_idx_list = [len(tmp_df)-1 for tmp_df in df_2x2_list]
    max_tree_idx_list = [len(tmp_df)-1 for tmp_df in df_tree_list]

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    def generate_individual():
        while True:
            individual = [random.randint(0, i) for i in max_2x2_idx_list] + [random.randint(0, j) for j in max_tree_idx_list]
            #T1 = time.perf_counter()
            ME = GetME(df_2x2_list, individual, mul_dist_list, tree_error_matrices_list)
            #T2 = time.perf_counter()
            #run_time = T2-T1
            #print(f"ME Time spent: {run_time:.4f} seconds")
            if ME_constraint[0] <= ME <= ME_constraint[1]:
                print([df_2x2_list[i].loc[individual[i],'mul_name'] for i in range(16)] + [df_tree_list[i].loc[individual[i+16],'tree_name'] for i in range(4)])
                return individual

    def evaluate(individual):
        estimated_area  = 0
        estimated_power = 0
        for i in range(16):
            estimated_area  += df_2x2_list[i].loc[individual[i],'area']
            estimated_power += df_2x2_list[i].loc[individual[i],'power']
        for i in range(4):
            estimated_area  += df_tree_list[i].loc[individual[i+16],'area']
            estimated_power += df_tree_list[i].loc[individual[i+16],'power']
        ME = GetME(df_2x2_list, individual, mul_dist_list, tree_error_matrices_list)
        if not (ME_constraint[0] <= ME <= ME_constraint[1]):
            ME = -5000000
            estimated_area = 500000000
            estimated_power = 500000000
        return estimated_area, estimated_power, np.abs(ME)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selNSGA2)
    
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0]*20, up=max_2x2_idx_list+max_tree_idx_list, indpb=0.3)

    population = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=2*pop_size, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=None, halloffame=None, verbose=True)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    return pareto_front
    

def Optimize8x8(pop_size=500, ngen=3000, cxpb=0.8, mutpb=0.2):
    loc_2x2_list = [''.join(bits) for bits in product('hl', repeat=4)]  # hhhh, hhhl, …, llll
    loc_4x4_list = ['hh', 'hl', 'lh', 'll']

    config_inst = Config()
    mul_dist_list = GetSubDist(config_inst.dist_file)

    # Load 2x2 multipliers
    df_2x2 = pd.read_csv('./out/lut_2x2.csv')
    df_2x2_list = []
    for loc in loc_2x2_list:
        tmp_df_2x2 = df_2x2.loc[(df_2x2.loc[:,'loc'] == loc)].copy()
        tmp_df_2x2 = tmp_df_2x2.sort_values(by='area').reset_index(drop=True)
        df_2x2_list.append(tmp_df_2x2)
    print(df_2x2_list)

    # Load 4-bit adder trees
    df_tree = pd.read_csv('./out/lut_tree.csv')
    df_tree_list = []
    tree_error_matrices_list = []
    for loc in loc_4x4_list:
        tmp_df_tree = df_tree.loc[(df_tree.loc[:,'loc'] == loc)].copy()
        tmp_df_tree = tmp_df_tree.sort_values(by='area').reset_index(drop=True)
        df_tree_list.append(tmp_df_tree)
        # Cache the error matrix for adder trees to speed up
        tmp_error_matrices = []
        for index, row in tmp_df_tree.iterrows():
            lut = json.loads(row['lut'])
            fa_lut_list = [lut[:8], lut[8:16], lut[16:24], lut[24:32]]
            rca_lut_list = [lut[32:40], lut[40:48], lut[48:56], lut[56:]]
            tmp_error_matrices.append(GetErrorMatrix4bit(row['tree_name'], fa_lut_list, rca_lut_list))
        tree_error_matrices_list.append(tmp_error_matrices)
    print(df_tree_list)

    # Run Optimization
    T1 = time.perf_counter()
    pareto_front = NSGA8x8(df_2x2_list, df_tree_list, tree_error_matrices_list, mul_dist_list, config_inst.ME_constraint, pop_size, ngen, cxpb, mutpb)
    T2 = time.perf_counter()
    run_time = T2-T1
    print(f"Total Time spent: {run_time:.4f} seconds")
    print('----------------------------------------')

    # Remove duplicate solutions
    unique_pareto_front = set(tuple(sublist) for sublist in pareto_front)
    unique_pareto_front = [list(sublist) for sublist in unique_pareto_front]
    
    # Save results
    df_8x8 = pd.DataFrame(columns = (['mul_name', 'structure', 'application', 'ME', 'area', 'power', 'gate_counts']))
    mul_8x8_idx = 0
    for index_list in unique_pareto_front:
        ME = GetME(df_2x2_list, index_list, mul_dist_list, tree_error_matrices_list)
        if not (config_inst.ME_constraint[0] <= ME <= config_inst.ME_constraint[1]): # Violations may occur due to mutations
            continue
        structure = [df_2x2_list[i].loc[index_list[i],'mul_name'] for i in range(16)] + [df_tree_list[i].loc[index_list[i+16],'tree_name'] for i in range(4)]
        structure = ','.join(structure)
        estimated_area  = 0
        estimated_power = 0
        gate_counts = 0
        for i in range(16):
            estimated_area  += df_2x2_list[i].loc[index_list[i],'area']
            estimated_power += df_2x2_list[i].loc[index_list[i],'power']
            gate_counts     += df_2x2_list[i].loc[index_list[i],'gate_counts']
        for i in range(4):
            estimated_area  += df_tree_list[i].loc[index_list[i+16],'area']
            estimated_power += df_tree_list[i].loc[index_list[i+16],'power']
            gate_counts     += df_tree_list[i].loc[index_list[i+16],'gate_counts']
        # Name the multiplier, the same structure has the same name
        same_structure_mul = df_8x8.loc[df_8x8.loc[:,'structure']==structure, 'mul_name']
        if structure == ','.join(['mul2_acc']*16) + ','.join(['tree_acc']*4):
            mul_8x8_name = 'mul8_acc'
        elif (same_structure_mul.empty):
            mul_8x8_name = 'mul8_' + str(mul_8x8_idx)
            mul_8x8_idx += 1
        else:
            mul_8x8_name = same_structure_mul.unique()[0]
        # Save
        new_row = [mul_8x8_name, structure, config_inst.application, ME, estimated_area, estimated_power, gate_counts]
        row_len = df_8x8.shape[0]
        df_8x8.loc[row_len] = new_row  # Add new_row to df
        print(new_row)

    df_8x8.to_csv('./out/mul_8x8.csv', index=False)
    print('Number of valid multipliers:', len(set(df_8x8.loc[:, 'mul_name'])))


if __name__ == "__main__":
    Optimize8x8(pop_size=500, ngen=3000, cxpb=0.8, mutpb=0.2)