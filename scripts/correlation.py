import numpy as np
import os

# Pi0 Stacking Data
instruction_stack_pi0 = [1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1]
success_stack_pi0 = [0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0]
instruction_stack_wm_pi0 = [1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1]
success_stack_wm_pi0 = [0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0]

# Pi0 Pick and Place Data
instruction_pick_and_place_pi0 = [1,0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,1,1,1]
success_pick_and_place_pi0 = [1,0,1,0,0,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1]
instruction_pick_and_place_wm_pi0 = [1,1,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,1,1,1]
success_pick_and_place_wm_pi0 = [1,1,1,0,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1]

# Pi0 Put in Drawer Data
instruction_drawer_pi0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
success_drawer_pi0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
instruction_drawer_wm_pi0 = [1,1,1,0,0,1,1,1,1,1,0,1,0,1,1,1]
success_drawer_wm_pi0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]

# Pi05 Stacking Data
instruction_stack_pi05 = [1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1]
success_stack_pi05 = [1,0,1,1,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,0]
instruction_stack_wm_pi05 = [1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1]
success_stack_wm_pi05 = [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0]

# Pi05 Pick and Place Data
instruction_pick_and_place_pi05 = [1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1]
success_pick_and_place_pi05 = [1,1,1,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1]
instruction_pick_and_place_wm_pi05 = [1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1]
success_pick_and_place_wm_pi05 = [1,1,1,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,1,1]

# Pi05 Put in Drawer Data
instruction_drawer_pi05 = [1,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,0]
success_drawer_pi05 = [1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,0,1,1,1,0]
instruction_drawer_pi05_wm = [1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0]
success_drawer_pi05_wm = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]

# Pifast stack blocks
instruction_stack_pifast = [1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
success_stack_pifast = [0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0]
instruction_stack_wm_pifast = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
success_stack_wm_pifast = [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Pifast pick and place blocks
instruction_pick_and_place_pifast = [1,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,1,1]
success_pick_and_place_pifast = [1,1,1,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,1]
instruction_pick_and_place_wm_pifast = [1,0,1,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1]
success_pick_and_place_wm_pifast = [1,0,1,1,0,1,0,0,1,1,0,0,0,1,1,0,0,0,1,1]

# Pifast put in drawer blocks
instruction_put_in_drawer_pifast = [0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,0,1,0,0]
success_put_in_drawer_pifast = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
instruction_put_in_drawer_wm_pifast = [1,1,0,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1]
success_put_in_drawer_wm_pifast = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Pi0 Correlation 
correlation_stack_instr_pi0 = np.corrcoef(instruction_stack_pi0, instruction_stack_wm_pi0)[0, 1]
correlation_stack_success_pi0 = np.corrcoef(success_stack_pi0, success_stack_wm_pi0)[0, 1]
correlation_pick_and_place_instr_pi0 = np.corrcoef(instruction_pick_and_place_pi0, instruction_pick_and_place_wm_pi0)[0, 1]
correlation_pick_and_place_success_pi0 = np.corrcoef(success_pick_and_place_pi0, success_pick_and_place_wm_pi0)[0, 1]
correlation_drawer_instr_pi0 = np.corrcoef(instruction_drawer_pi0, instruction_drawer_wm_pi0)[0, 1]
correlation_drawer_success_pi0 = np.corrcoef(success_drawer_pi0, success_drawer_wm_pi0)[0, 1]   

print("Pi0 Correlations:")
print("Stacking Instruction Correlation:", correlation_stack_instr_pi0)
print("Stacking Success Correlation:", correlation_stack_success_pi0)
print("Pick and Place Instruction Correlation:", correlation_pick_and_place_instr_pi0)
print("Pick and Place Success Correlation:", correlation_pick_and_place_success_pi0)
print("Put in Drawer Instruction Correlation:", correlation_drawer_instr_pi0)
print("Put in Drawer Success Correlation:", correlation_drawer_success_pi0)
print("\n")

# Pi05 Correlations
correlation_stack_instr_pi05 = np.corrcoef(instruction_stack_pi05, instruction_stack_wm_pi05)[0, 1]
correlation_stack_success_pi05 = np.corrcoef(success_stack_pi05, success_stack_wm_pi05)[0, 1]
correlation_pick_and_place_instr_pi05 = np.corrcoef(instruction_pick_and_place_pi05, instruction_pick_and_place_wm_pi05)[0, 1]
correlation_pick_and_place_success_pi05 = np.corrcoef(success_pick_and_place_pi05, success_pick_and_place_wm_pi05)[0, 1]
correlation_drawer_instr_pi05 = np.corrcoef(instruction_drawer_pi05, instruction_drawer_pi05_wm)[0, 1]
correlation_drawer_success_pi05 = np.corrcoef(success_drawer_pi05, success_drawer_pi05_wm)[0, 1]   

print("\nPi05 Correlations:")
print("Stacking Instruction Correlation:", correlation_stack_instr_pi05)
print("Stacking Success Correlation:", correlation_stack_success_pi05)
print("Pick and Place Instruction Correlation:", correlation_pick_and_place_instr_pi05)
print("Pick and Place Success Correlation:", correlation_pick_and_place_success_pi05)
print("Put in Drawer Instruction Correlation:", correlation_drawer_instr_pi05)
print("Put in Drawer Success Correlation:", correlation_drawer_success_pi05)
print("\n")

# Pifast Correlations
correlation_stack_instr_pifast = np.corrcoef(instruction_stack_pifast, instruction_stack_wm_pifast)[0, 1]
correlation_stack_success_pifast = np.corrcoef(success_stack_pifast, success_stack_wm_pifast)[0, 1]
correlation_pick_and_place_instr_pifast = np.corrcoef(instruction_pick_and_place_pifast, instruction_pick_and_place_wm_pifast)[0,1]
correlation_pick_and_place_success_pifast = np.corrcoef(success_pick_and_place_pifast, success_pick_and_place_wm_pifast)[0, 1]
correlation_drawer_instr_pifast = np.corrcoef(instruction_put_in_drawer_pifast, instruction_put_in_drawer_wm_pifast)[0, 1]
correlation_drawer_success_pifast = np.corrcoef(success_put_in_drawer_pifast, success_put_in_drawer_wm_pifast)[0, 1]

print("\nPifast Correlations:")
print("Stacking Instruction Correlation:", correlation_stack_instr_pifast)
print("Stacking Success Correlation:", correlation_stack_success_pifast)
print("Pick and Place Instruction Correlation:", correlation_pick_and_place_instr_pifast)
print("Pick and Place Success Correlation:", correlation_pick_and_place_success_pifast)
print("Put in Drawer Instruction Correlation:", correlation_drawer_instr_pifast)
print("Put in Drawer Success Correlation:", correlation_drawer_success_pifast)
print("\n")