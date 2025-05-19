import os
import pandas as pd
from config import Config


def GetPareto(df, target_hardware_cost):
    """
    Use traversal method to find the pareto front
    """
    df_sorted = df.sort_values(by='accuracy', ascending=False)
    pareto_frontier = []
    for index, row in df_sorted.iterrows():
        if not pareto_frontier:
            pareto_frontier.append(row.to_dict())
        else:
            if len(target_hardware_cost) == 2:
                min_metric0 = True
                min_metric1 = True
                for pareto_solution in pareto_frontier:
                    if row[target_hardware_cost[0]] > pareto_solution[target_hardware_cost[0]]:
                        min_metric0 = False
                    if row[target_hardware_cost[1]] > pareto_solution[target_hardware_cost[1]]:
                        min_metric1 = False
                if min_metric0 or min_metric1:
                    pareto_frontier.append(row.to_dict())
            else:
                min_metric = True
                for pareto_solution in pareto_frontier:
                    if row[target_hardware_cost[0]] > pareto_solution[target_hardware_cost[0]]:
                        min_metric = False
                if min_metric:
                    pareto_frontier.append(row.to_dict())

    df_pareto = pd.DataFrame(pareto_frontier)
    return df_pareto


def FinalPareto(target_hardware_cost, out_path):
    """
    Construct the final Pareto front based on metrics obtained from simulation and synthesis
    """
    config_inst = Config()
    df_8x8 = pd.read_csv('./out/mul_8x8.csv')
    df_8x8.rename(columns={'area' : 'Area', 'power' : 'Power', 'delay' : 'Delay'}, inplace=True)
    
    if 'PDP' not in df_8x8.columns:
        df_8x8['PDP'] = df_8x8['Power'] * df_8x8['Delay']

    if 'ADP' not in df_8x8.columns:
        df_8x8['ADP'] = df_8x8['Area'] * df_8x8['Delay']

    df_8x8 = df_8x8.loc[df_8x8.loc[:, 'accuracy'] >= config_inst.accuracy_threshold, :].copy() 
    df_pareto = GetPareto(df_8x8, target_hardware_cost)
    out_file = os.path.join(out_path, 'mul_8x8_pareto.csv')
    df_pareto.to_csv(out_file, index=False)
    print(f"[FINISH] Final Pareto-optimal solutions have been written to {out_file}")


if __name__ == '__main__':
    #target_hardware_cost = ['Area'] # 1 or 2 metrics
    target_hardware_cost = ['Power', 'Area'] # 1 or 2 metrics

    FinalPareto(target_hardware_cost, './out')
