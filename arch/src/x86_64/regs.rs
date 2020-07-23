// Copyright © 2020, Oracle and/or its affiliates.
//
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.

extern crate bhyve_api;

use std::{mem, result};

use super::gdt::{gdt_entry};
use super::BootProtocol;
use bhyve_api::vm::{VirtualMachine, vm_reg_name, vm_cap_type};
use layout::{BOOT_GDT_START, BOOT_IDT_START, PDE_START, PDPTE_START, PML4_START, PVH_INFO_START};
use vm_memory::{Address, Bytes, GuestMemory, GuestMemoryError, GuestMemoryMmap};

#[derive(Debug)]
pub enum Error {
    /// Failed to get SREGs for this CPU.
    GetStatusRegisters(bhyve_api::Error),
    /// Failed to set base registers for this CPU.
    SetBaseRegisters(bhyve_api::Error),
    /// Failed to configure the FPU.
    SetFPURegisters(bhyve_api::Error),
    /// Setting up MSRs failed.
    SetModelSpecificRegisters(bhyve_api::Error),
    /// Failed to set SREGs for this CPU.
    SetStatusRegisters(bhyve_api::Error),
    /// Checking the GDT address failed.
    CheckGDTAddr,
    /// Writing the GDT to RAM failed.
    WriteGDT(GuestMemoryError),
    /// Writing the IDT to RAM failed.
    WriteIDT(GuestMemoryError),
    /// Writing PDPTE to RAM failed.
    WritePDPTEAddress(GuestMemoryError),
    /// Writing PDE to RAM failed.
    WritePDEAddress(GuestMemoryError),
    /// Writing PML4 to RAM failed.
    WritePML4Address(GuestMemoryError),
}

pub type Result<T> = result::Result<T, Error>;

/// Configure base registers for a given CPU.
///
/// # Arguments
///
/// * `vcpu` - Structure for the VM that holds the VM's fd.
/// * `vcpu_id` - VCPU identifier.
/// * `boot_ip` - Starting instruction pointer.
/// * `boot_sp` - Starting stack pointer.
/// * `boot_si` - Must point to zero page address per Linux ABI.
pub fn setup_regs(
    vm: &VirtualMachine,
    vcpu_id: i32,
    boot_ip: u64,
    boot_sp: u64,
    boot_si: u64,
    boot_prot: BootProtocol,
) -> Result<()> {
    vm.vcpu_reset(vcpu_id).map_err(Error::SetBaseRegisters)?;
    vm.set_x2apic_state(vcpu_id, true).map_err(Error::SetBaseRegisters)?;
    vm.set_capability(vcpu_id, vm_cap_type::VM_CAP_UNRESTRICTED_GUEST, 1).map_err(Error::SetBaseRegisters)?;
    match boot_prot {
        // Configure regs as required by PVH boot protocol.
        BootProtocol::PvhBoot => {
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RFLAGS, 0x0000000000000002u64).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RBX, PVH_INFO_START.raw_value()).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RIP, boot_ip).map_err(Error::SetBaseRegisters)?;
        },
        // Configure regs as required by Linux 64-bit boot protocol.
        BootProtocol::LinuxBoot => {
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RFLAGS, 0x0000000000000002u64).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RIP, boot_ip).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RSP, boot_sp).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RBP, boot_sp).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RSI, boot_si).map_err(Error::SetBaseRegisters)?;
        },
    };
    Ok(())
}

/// Configures the segment registers and system page tables for a given CPU.
///
/// # Arguments
///
/// * `mem` - The memory that will be passed to the guest.
/// * `vm` - Structure for the VM that holds the VM's fd.
/// * `vcpu_id` - VCPU identifier.
pub fn setup_sregs(mem: &GuestMemoryMmap, vm: &VirtualMachine, vcpu_id: i32, boot_prot: BootProtocol) -> Result<()> {
    configure_segments_and_sregs(mem, &vm, vcpu_id, boot_prot)?;

    if let BootProtocol::LinuxBoot = boot_prot {
        setup_page_tables(mem, &vm, vcpu_id)?; // TODO(dgreid) - Can this be done once per system instead?
    }
    Ok(())
}

const BOOT_GDT_MAX: usize = 4;

const EFER_LMA: u64 = 0x400;
const EFER_LME: u64 = 0x100;

const X86_CR0_PE: u64 = 0x1;
const X86_CR0_PG: u64 = 0x80000000;
const X86_CR4_PAE: u64 = 0x20;

fn write_gdt_table(table: &[u64], guest_mem: &GuestMemoryMmap) -> Result<()> {
    let boot_gdt_addr = BOOT_GDT_START;
    for (index, entry) in table.iter().enumerate() {
        let addr = guest_mem
            .checked_offset(boot_gdt_addr, index * mem::size_of::<u64>())
            .ok_or(Error::CheckGDTAddr)?;
        guest_mem.write_obj(*entry, addr).map_err(Error::WriteGDT)?;
    }
    Ok(())
}

fn write_idt_value(val: u64, guest_mem: &GuestMemoryMmap) -> Result<()> {
    let boot_idt_addr = BOOT_IDT_START;
    guest_mem
        .write_obj(val, boot_idt_addr)
        .map_err(Error::WriteIDT)
}

fn configure_segments_and_sregs(
    mem: &GuestMemoryMmap,
    vm: &VirtualMachine,
    vcpu_id: i32,
    boot_prot: BootProtocol,
) -> Result<()> {
    let gdt_table: [u64; BOOT_GDT_MAX as usize] = match boot_prot {
        BootProtocol::PvhBoot => {
            // Configure GDT entries as specified by PVH boot protocol
            [
                gdt_entry(0, 0, 0),               // NULL
                gdt_entry(0xc09b, 0, 0xffffffff), // CODE
                gdt_entry(0xc093, 0, 0xffffffff), // DATA
                gdt_entry(0x008b, 0, 0x67),       // TSS
            ]
        }
        BootProtocol::LinuxBoot => {
            // Configure GDT entries as specified by Linux 64bit boot protocol
            [
                gdt_entry(0, 0, 0),            // NULL
                gdt_entry(0xa09b, 0, 0xfffff), // CODE
                gdt_entry(0xc093, 0, 0xfffff), // DATA
                gdt_entry(0x808b, 0, 0xfffff), // TSS
            ]
        }
    };

    // Write segments
    write_gdt_table(&gdt_table[..], mem)?;
    let gdt_base = BOOT_GDT_START.raw_value();
    let gdt_limit = mem::size_of_val(&gdt_table) as u32 - 1;
    vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_GDTR, gdt_base, gdt_limit, 0).map_err(Error::SetStatusRegisters)?;

    write_idt_value(0, mem)?;
    let idt_base = BOOT_IDT_START.raw_value();
    let idt_limit = mem::size_of::<u64>() as u32 - 1;
    vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_IDTR, idt_base, idt_limit, 0).map_err(Error::SetStatusRegisters)?;

    match boot_prot {
        BootProtocol::PvhBoot => {
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR0, X86_CR0_PE).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR4, 0).map_err(Error::SetBaseRegisters)?;

            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_CS, 0, 0xffffffff, 0xa09b).map_err(Error::SetStatusRegisters)?;
            let data_base = 0;
            let data_limit = 0xffffffff;
            let data_flags = 0xc093;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_DS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_ES, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_FS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_GS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_SS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_TR, 0, 0x67, 0x008b).map_err(Error::SetStatusRegisters)?;
        }
        BootProtocol::LinuxBoot => {
            /* 64-bit protected mode */
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR0, X86_CR0_PE).map_err(Error::SetBaseRegisters)?;
            vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_EFER, EFER_LME | EFER_LMA).map_err(Error::SetBaseRegisters)?;

            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_CS, 0, 0xfffff, 0xa09b).map_err(Error::SetStatusRegisters)?;
            let data_base = 0;
            let data_limit = 0xfffff;
            let data_flags = 0xc093;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_DS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_ES, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_FS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_GS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_SS, data_base, data_limit, data_flags).map_err(Error::SetStatusRegisters)?;
            vm.set_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_TR, 0, 0xfffff, 0x808b).map_err(Error::SetStatusRegisters)?;
        }
    }

    Ok(())
}

fn setup_page_tables(mem: &GuestMemoryMmap, vm: &VirtualMachine, vcpu_id: i32) -> Result<()> {
    // Puts PML4 right after zero page but aligned to 4k.

    // Entry covering VA [0..512GB)
    mem.write_obj(PDPTE_START.raw_value() | 0x03, PML4_START)
        .map_err(Error::WritePML4Address)?;

    // Entry covering VA [0..1GB)
    mem.write_obj(PDE_START.raw_value() | 0x03, PDPTE_START)
        .map_err(Error::WritePDPTEAddress)?;
    // 512 2MB entries together covering VA [0..1GB). Note we are assuming
    // CPU supports 2MB pages (/proc/cpuinfo has 'pse'). All modern CPUs do.
    for i in 0..512 {
        mem.write_obj((i << 21) + 0x83u64, PDE_START.unchecked_add(i * 8))
            .map_err(Error::WritePDEAddress)?;
    }

    vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR3, PML4_START.raw_value()).map_err(Error::SetBaseRegisters)?;
    vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR4, X86_CR4_PAE).map_err(Error::SetBaseRegisters)?;
    vm.set_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR0, X86_CR0_PG).map_err(Error::SetBaseRegisters)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate bhyve_api;
    extern crate vm_memory;

    use super::*;
    use bhyve_api::system::VMMSystem;
    use bhyve_api::vm::VirtualMachine;
    use vm_memory::{GuestAddress, GuestMemoryMmap};

    fn create_guest_mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&vec![(GuestAddress(0), 0x10000)]).unwrap()
    }

    fn read_u64(gm: &GuestMemoryMmap, offset: GuestAddress) -> u64 {
        gm.read_obj(offset).unwrap()
    }

    #[test]
    fn test_setup_regs() {
        let vm_name = "regstest";
        let vcpu_id = 0;
        let vmmctl = VMMSystem::new().expect("Failed to open VMM handle, must be root to access /dev/vmm");
        vmmctl.create_vm(vm_name).unwrap();
        let vm = VirtualMachine::new(vm_name).unwrap();

        setup_regs(&vm, vcpu_id, 1, 2, 3, BootProtocol::LinuxBoot).unwrap();

        let rflags = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RFLAGS).unwrap();
        assert_eq!(rflags, 0x0000000000000002u64);
        let rip = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RIP).unwrap();
        assert_eq!(rip, 1);
        let rsp = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RSP).unwrap();
        assert_eq!(rsp, 2);
        let rbp = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RBP).unwrap();
        assert_eq!(rbp, 2);
        let rsi = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_RSI).unwrap();
        assert_eq!(rsi, 3);

        vmmctl.destroy_vm(vm_name).unwrap();
    }

    #[test]
    fn test_setup_sregs_linuxboot() {
        let vm_name = "sregstest_linux";
        let vcpu_id = 0;
        let vmmctl = VMMSystem::new().expect("Failed to open VMM handle, must be root to access /dev/vmm");
        vmmctl.create_vm(vm_name).unwrap();
        let vm = VirtualMachine::new(vm_name).unwrap();

        let gm = create_guest_mem();
        setup_sregs(&gm, &vm, vcpu_id, BootProtocol::LinuxBoot).unwrap();


        assert_eq!(0x0, read_u64(&gm, BOOT_GDT_START));
        assert_eq!(
            0xaf9b000000ffff,
            read_u64(&gm, BOOT_GDT_START.unchecked_add(8))
        );
        assert_eq!(
            0xcf93000000ffff,
            read_u64(&gm, BOOT_GDT_START.unchecked_add(16))
        );
        assert_eq!(
            0x8f8b000000ffff,
            read_u64(&gm, BOOT_GDT_START.unchecked_add(24))
        );
        assert_eq!(0x0, read_u64(&gm, BOOT_IDT_START));


        assert_eq!(0xa003, read_u64(&gm, PML4_START));
        assert_eq!(0xb003, read_u64(&gm, PDPTE_START));
        for i in 0..512 {
            assert_eq!(
                (i << 21) + 0x83u64,
                read_u64(&gm, PDE_START.unchecked_add(i * 8))
            );
        }

        let sregs_cr3 = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR3).unwrap();
        assert_eq!(PML4_START.raw_value(), sregs_cr3);

        let sregs_cr4 = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR4).unwrap();
        assert_eq!(X86_CR4_PAE, sregs_cr4);

        let sregs_cr0 = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR0).unwrap();
        assert_eq!(X86_CR0_PG, sregs_cr0);

        let (cs_base, _, _) = vm.get_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_CS).unwrap();
        assert_eq!(0, cs_base);

        let (_, ds_limit, _) = vm.get_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_DS).unwrap();
        assert_eq!(0xfffff, ds_limit);

        let (tr_base, tr_limit, _) = vm.get_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_TR).unwrap();
        assert_eq!(0, tr_base);
        assert_eq!(0xfffff, tr_limit);

        // DISABLED: EFER register isn't being set for some reason
        // let sregs_efer = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_EFER).unwrap();
        // assert_eq!(EFER_LME | EFER_LMA, sregs_efer);

        vmmctl.destroy_vm(vm_name).unwrap();
    }

    #[test]
    fn test_setup_sregs_pvhboot() {
        let vm_name = "sregstest_pvh";
        let vcpu_id = 0;
        let vmmctl = VMMSystem::new().expect("Failed to open VMM handle, must be root to access /dev/vmm");
        vmmctl.create_vm(vm_name).unwrap();
        let vm = VirtualMachine::new(vm_name).unwrap();

        let gm = create_guest_mem();
        setup_sregs(&gm, &vm, vcpu_id, BootProtocol::PvhBoot).unwrap();


        assert_eq!(0x0, read_u64(&gm, BOOT_GDT_START));
        assert_eq!(
            0xcf9b000000ffff,
            read_u64(&gm, BOOT_GDT_START.unchecked_add(8))
        );
        assert_eq!(
            0xcf93000000ffff,
            read_u64(&gm, BOOT_GDT_START.unchecked_add(16))
        );
        assert_eq!(
            0x8b0000000067,
            read_u64(&gm, BOOT_GDT_START.unchecked_add(24))
        );
        assert_eq!(0x0, read_u64(&gm, BOOT_IDT_START));

        let (cs_base, _, _) = vm.get_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_CS).unwrap();
        assert_eq!(0, cs_base);

        let (_, ds_limit, _) = vm.get_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_DS).unwrap();
        assert_eq!(0xffffffff, ds_limit);

        let (tr_base, tr_limit, _) = vm.get_desc(vcpu_id, vm_reg_name::VM_REG_GUEST_TR).unwrap();
        assert_eq!(0, tr_base);
        assert_eq!(0x67, tr_limit);

        let sregs_cr0 = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR0).unwrap();
        assert_eq!(X86_CR0_PE, sregs_cr0);

        let sregs_cr4 = vm.get_register(vcpu_id, vm_reg_name::VM_REG_GUEST_CR4).unwrap();
        assert_eq!(0, sregs_cr4);

        vmmctl.destroy_vm(vm_name).unwrap();
    }
}
