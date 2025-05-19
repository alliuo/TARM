# TARM

TARM is an open-source framework that allows the synthesis of approximate recursive multipliers tailored to specific CNN applications.

## Dependencies

Refer to [dependencies.txt](dependencies.txt) for the necessary dependencies (you can modify the desired versions, as there are no strict version limitations). The default tools include Synopsys Design Compiler (DC) and VCS. If these tools are not available, you need to modify the synthesis-related codes within TARM.

The [ABC](https://github.com/berkeley-abc/abc) and [TFApprox (GPU version)](https://github.com/ehw-fit/tf-approximate) are required. After installation, you need to modified the paths in [config.yml](TARM/config/config.yml).

## Project Structure
```text
TARM/
├── config/                        # Configuration
│   ├── config.yaml                # Primary configuration file: paths, application, constraints
│   ├── dc_step1_8x8_template.tcl  # DC script template for 8x8 multpipliers (run sythesis)
|   ├── dc_step1_template.tcl      # DC script template for building blocks (run sythesis)
|   ├── dc_step2_template.tcl      # DC script template (reports hardware costs)
│   ├── makefile.template          # VCS makefile template
│   └── adder_tree_8bit/           # Verilog codes of the exact 8-bit adder tree
│       ├── final_tree_acc.v
│       └── ... 
├── config.py                      # Config loader
├── optimize_2x2.py                # Optimization of 2×2 multipliers
├── optimize_tree.py               # Optimization of 4-bit adder trees
├── run_opt_2x2_tree.py            # Parallel optimization of building blocks
├── build_syn_env_2x2_tree.py      # Build a synthesis environment for building blocks
├── run_syn.py                     # Synthesis
├── optimize_8x8.py                # Optimization of 8x8 multipliers
├── build_syn_env_8x8.py           # Build synthesis and evaluation environments for 8x8 multpipliers
├── final_pareto.py                # Identify Pareto-optimal 8x8 multpipliers
├── run_eval.py                    # Accuracy evaluation of AM-based CNN applications
└── main.py                        # Project overview and quickstart
```

## Running an Example

The folder [example](example/) provides ResNet-18 on CIFAR-10 as an example application. After modified the [config.yml](TARM/config/config.yml), running the example as follows. With the default configuration, the execution may take several hours.

```bash
cd TARM
python main.py
```

