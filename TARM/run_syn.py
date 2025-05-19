import os
import shutil
import subprocess
import pandas as pd


def VerifySAIF(saif_report):
    """
    Analyze saif report and verify the usr annotation rate
    """
    with open(saif_report, 'r') as f:
        lines = f.readlines()
    if len(lines) >= 5:
        usr_annotated_nets = lines[-5].strip().split()[1]
        usr_annotated_ports = lines[-4].strip().split()[1]
        usr_annotated_pins = lines[-3].strip().split()[1]
    
    if(usr_annotated_nets.endswith('(100.00%)') and usr_annotated_ports.endswith('(100.00%)') and usr_annotated_pins.endswith('(100.00%)')):
        return True
    else:
        return False


def ExtractPower(power_report):
    """
    Analyze power report and extract total power
    """
    with open(power_report, 'r') as f:
        lines = f.readlines()
    if len(lines) >= 2:
        second_to_last_line = lines[-2].strip().split()
    
    value = float(second_to_last_line[-2])
    unit  = second_to_last_line[-1]
    # Unify the units to uW
    if(unit == 'W'):
        power = value * 1e6
    elif(unit == 'mW'):
        power = value * 1e3
    elif(unit == 'uW'):
        power = value
    elif(unit == 'nW'):
        power = value * 1e-3
    elif(unit == 'pW'):
        power = value * 1e-6
    else:
        power = 0

    return power


def ExtractArea(area_report):
    """
    Analyze area report and extract area
    """
    with open(area_report, 'r') as f:
        area_report_content = f.read()

    area = None
    for line in area_report_content.split('\n'):
        if "Total cell area:" in line:
            area = float(line.split()[-1])
            return area
    return None


def ExtractDelay(timing_report):
    """
    Analyze timing report and extract critical path delay
    """
    with open(timing_report, 'r') as f:
        timing_report_content = f.read()
        
    delay = None
    for line in timing_report_content.split('\n'):
        if "data arrival time" in line:
            delay = float(line.split()[-1])
            return delay
    return 0
        

def ExtractSynResult(base_dir):
    """
    Analyze reports and extract key information
    """
    rtl_dir = os.path.join(base_dir, 'rtl')
    out_dir = os.path.join(base_dir, 'out')
    result_file = os.path.join(base_dir, 'syn_result.csv')

    rtl_list = os.listdir(rtl_dir)

    with open(result_file, 'w') as f:
        f.write('name,power,area,delay\n')
        for rtl in rtl_list:
            name = rtl[:-2] # remove '.v'
            saif_report   = os.path.join(out_dir, f'{name}.saif.rpts')
            power_report  = os.path.join(out_dir, f'{name}.power.rpts')
            area_report   = os.path.join(out_dir, f'{name}.area.rpts')
            timing_report = os.path.join(out_dir, f'{name}.timing.rpts')

            if(VerifySAIF(saif_report)):
                power = ExtractPower(power_report)
                area  = ExtractArea(area_report)
                delay = ExtractDelay(timing_report)
            else:
                print('Error:', name, ' not 100%% annotated')
                power = None
                area  = None
                delay = None

            new_line = name + ',' + str(power) + ',' + str(area) + ',' + str(delay) + '\n'
            f.write(new_line)


def RunSynthesis(base_dir):
    """
    Start a subprocess to execute the synthesis
    """
    base_dir = os.path.abspath(base_dir)

    rtl_dir = os.path.join(base_dir, 'rtl')
    rtl_list = os.listdir(rtl_dir)
    unit_name_list = [rtl[:-2] for rtl in rtl_list] # remove '.v'

    mapped_dir = os.path.join(base_dir, 'mapped')
    if os.path.exists(mapped_dir):
        shutil.rmtree(mapped_dir)
    os.makedirs(mapped_dir)

    out_dir = os.path.join(base_dir, 'out')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    # DC step 1 (generate netlist)
    for unit in unit_name_list:
        run_dir = os.path.join(base_dir, 'run')
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        os.chdir(run_dir) # cd /run
        cmd = ['dc_shell', '-f', os.path.join('..', 'dc_step1', f'{unit}_step1.tcl')]
        print(f"[DC STEP1] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        os.chdir(base_dir) # cd ..
    
    # VCS simulation (generate the back-annotated switching activity file)
    vcs_dir = os.path.join(base_dir, 'vcs')
    os.chdir(vcs_dir)  # cd /vcs
    print(f"[VCS] Running simulation in {vcs_dir}")
    subprocess.run(['make', 'regress'], check=True)
    os.chdir(base_dir) # cd ..
    
    # DC step 2 (report)
    for unit in unit_name_list:
        run_dir = os.path.join(base_dir, 'run')
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        os.chdir(run_dir)  # cd /run
        cmd = ['dc_shell', '-f', os.path.join('..', 'dc_step2', f'{unit}_step2.tcl')]
        print(f"[DC STEP2] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        os.chdir(base_dir) # cd ..
    
    # Extract synthesis result
    print(f"[EXTRACT] Extract synthesis results, save in {base_dir}")
    ExtractSynResult(base_dir)


def MergeSynResult(syn_dir, out_file):
    """
    Merge the synthesis result with other metrics
    """
    syn_result_file = os.path.join(syn_dir, 'syn_result.csv')
    df_syn = pd.read_csv(syn_result_file)
    df_out = pd.read_csv(out_file)
    if 'power' in df_out.columns:
        return
    if 'mul_name' in df_out.columns:
        df_syn = df_syn.rename(columns={'name': 'mul_name'})
        merged_df = pd.merge(df_out, df_syn, on='mul_name')
    elif 'tree_name' in df_out.columns:
        df_syn = df_syn.rename(columns={'name': 'tree_name'})
        merged_df = pd.merge(df_out, df_syn, on='tree_name')
    merged_df.to_csv(out_file, index=False)
