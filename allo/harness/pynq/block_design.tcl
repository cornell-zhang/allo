# Hardcoded block design script (only for single vvadd kernel)

# create project named vvadd, in the directory build_vivado
create_project vvadd ./build_vivado -part xczu3eg-sbva484-1-i

# ADD IP REPOSITORIES
# create sources_1 fileset if it does not exist
if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}

# update the sources_1 fileset's ip repo path property to be the location of vvadd ip
set_property ip_repo_paths "out.prj/solution1/impl/ip" [get_filesets sources_1]
# make ip available in ip catalog so you can instantiate ip in bd or tcl
update_ip_catalog -rebuild

# CREATE BLOCK DESIGN
create_bd_design "vvadd_bd"

# add components
create_bd_cell -type ip -vlnv xilinx.com:hls:vvadd:1.0 vvadd_0

set clk_wiz_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0]
set_property CONFIG.CLKOUT1_REQUESTED_OUT_FREQ 200 $clk_wiz_0
set_property CONFIG.RESET_PORT {resetn} $clk_wiz_0
set_property CONFIG.RESET_TYPE {ACTIVE_LOW} $clk_wiz_0

set zynq_ultra_ps_e_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]
set_property CONFIG.PSU__USE__S_AXI_GP0 1 $zynq_ultra_ps_e_0
set_property CONFIG.PSU__SAXIGP0__DATA_WIDTH 32 $zynq_ultra_ps_e_0

set ctrl_axi_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 ctrl_axi_smc]
set data_axi_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 data_axi_smc]

# CONNECT AXI SYSTEM
set_property CONFIG.NUM_SI 1 $data_axi_smc
set_property CONFIG.NUM_MI 1 $data_axi_smc
set_property CONFIG.NUM_SI 1 $ctrl_axi_smc
set_property CONFIG.NUM_MI 1 $ctrl_axi_smc

# AXI connections
# for the data AXI network
connect_bd_intf_net [get_bd_intf_pins vvadd_0/m_axi_gmem] [get_bd_intf_pins data_axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins data_axi_smc/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

# for the control AXI network
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins ctrl_axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl_axi_smc/M00_AXI] [get_bd_intf_pins vvadd_0/s_axi_control]

# Address mapping
# for data network
assign_bd_address -offset 0x00000000 -range 2G -target_address_space [get_bd_addr_spaces vvadd_0/Data_m_axi_gmem] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW]

# for control network
assign_bd_address -offset 0x80000000 -range 4K -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs vvadd_0/s_axi_control/Reg]

# Clock & Resets
# Processor & vvadd kernel clocks
# 100 MHz domain
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins clk_wiz_0/clk_in1] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk] [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]
# 200 MHz domain
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins vvadd_0/ap_clk]

# config smartconnects to accept 2 clocks & connect them
set_property CONFIG.NUM_CLKS 2 $data_axi_smc
set_property CONFIG.NUM_CLKS 2 $ctrl_axi_smc

connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins data_axi_smc/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins ctrl_axi_smc/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins data_axi_smc/aclk1]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins ctrl_axi_smc/aclk1]

# Reset signals
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_200M
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins reset_200M/ext_reset_in] [get_bd_pins clk_wiz_0/resetn]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins reset_200M/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins reset_200M/dcm_locked]
connect_bd_net [get_bd_pins reset_200M/peripheral_aresetn] [get_bd_pins vvadd_0/ap_rst_n] [get_bd_pins data_axi_smc/aresetn] [get_bd_pins ctrl_axi_smc/aresetn]

# VALIDATE BD & CREATE RTL WRAPPER
validate_bd_design
save_bd_design
close_bd_design vvadd_bd

set_property REGISTERED_WITH_MANAGER "1" [get_files vvadd_bd.bd]
set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [get_files vvadd_bd.bd]
set wrapper_file [make_wrapper -fileset sources_1 -files [get_files vvadd_bd.bd] -top]

# add wrapper file to project & set it as top-level module
add_files -norecurse -fileset [get_filesets sources_1] $wrapper_file
set_property top vvadd_bd_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1

# SETUP SYNTHESIS & IMPLEMENTATION RUNS
# synthesis run "synth_1"
if {[string equal [get_runs -quiet synth_1] ""]} {
    create_run -flow "Vivado Synthesis 2023" -name synth_1 -part xczu3eg-sbva484-1-i
}
set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1]
set_property report_strategy "Vivado Synthesis Default Reports" [get_runs synth_1]

# implementation run "impl_1"
# it's a child run of "synth_1" since implementation depends on synthesis
if {[string equal [get_runs -quiet impl_1] ""]} {
    create_run -flow "Vivado Implementation 2023" -name impl_1 -part xczu3eg-sbva484-1-i -parent_run [get_runs synth_1]
}
set_property strategy "Vivado Implementation Defaults" [get_runs impl_1]
set_property report_strategy "Vivado Implementation Default Reports" [get_runs impl_1]

# LAUNCH SYNTHESIS & IMPLEMENTATION RUNS
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

exit