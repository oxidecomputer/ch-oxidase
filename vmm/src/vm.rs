// Copyright © 2020, Oracle and/or its affiliates.
//
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE-BSD-3-Clause file.
//
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//

extern crate arch;
extern crate devices;
extern crate epoll;
extern crate libc;
extern crate linux_loader;
extern crate signal_hook;
extern crate vm_allocator;
extern crate vm_memory;
extern crate vm_virtio;
extern crate bhyve_api;

use crate::config::{
    DeviceConfig, DiskConfig, FsConfig, HotplugMethod, NetConfig, PmemConfig, ValidationError,
    VmConfig, VsockConfig,
};
use crate::cpu;
use crate::device_manager::{get_win_size, Console, DeviceManager, DeviceManagerError};
use crate::memory_manager::{Error as MemoryManagerError, MemoryManager};
use arch::{BootProtocol, EntryPoint};
use devices::{ioapic, HotPlugNotificationFlags};
use bhyve_api::system::VMMSystem;
use bhyve_api::vm::*;
use linux_loader::cmdline::Cmdline;
use linux_loader::loader::elf::Error::InvalidElfMagicNumber;
use linux_loader::loader::KernelLoader;
use signal_hook::{iterator::Signals, SIGINT, SIGTERM, SIGWINCH};
use std::convert::TryInto;
use std::ffi::CString;
use std::fs::File;
use std::io::{self, Seek, SeekFrom};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::{result, str, thread};
use vm_memory::{
    Address, Bytes, GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryMmap,
    GuestMemoryRegion, MemoryRegionAddress,
};
use vmm_sys_util::eventfd::EventFd;
use vmm_sys_util::terminal::Terminal;

// 64 bit direct boot entry offset for bzImage
const KERNEL_64BIT_ENTRY_OFFSET: u64 = 0x200;

/// Errors associated with VM management
#[derive(Debug)]
pub enum Error {
    /// Cannot open the VM file descriptor.
    VmFd(io::Error),

    /// Cannot create the VM instance
    VmCreate(bhyve_api::Error),

    /// Cannot set the VM up
    VmSetup(bhyve_api::Error),

    /// Cannot open the kernel image
    KernelFile(io::Error),

    /// Cannot open the initramfs image
    InitramfsFile(io::Error),

    /// Cannot load the kernel in memory
    KernelLoad(linux_loader::loader::Error),

    /// Cannot load the initramfs in memory
    InitramfsLoad,

    /// Cannot load the command line in memory
    LoadCmdLine(linux_loader::loader::Error),

    /// Cannot modify the command line
    CmdLineInsertStr(linux_loader::cmdline::Error),

    /// Cannot convert command line into CString
    CmdLineCString(std::ffi::NulError),

    /// Cannot configure system
    ConfigureSystem(arch::Error),

    PoisonedState,

    /// Cannot create a device manager.
    DeviceManager(DeviceManagerError),

    /// Write to the console failed.
    Console(vmm_sys_util::errno::Error),

    /// Cannot setup terminal in raw mode.
    SetTerminalRaw(vmm_sys_util::errno::Error),

    /// Cannot setup terminal in canonical mode.
    SetTerminalCanon(vmm_sys_util::errno::Error),

    /// Failed parsing network parameters
    ParseNetworkParameters,

    /// Memory is overflow
    MemOverflow,

    /// Failed to allocate the IOAPIC memory range.
    IoapicRangeAllocation,

    /// Cannot spawn a signal handler thread
    SignalHandlerSpawn(io::Error),

    /// Failed to join on vCPU threads
    ThreadCleanup(std::boxed::Box<dyn std::any::Any + std::marker::Send>),

    /// Failed to create a new Bhyve instance
    BhyveNew(bhyve_api::Error),

    /// VM is not created
    VmNotCreated,

    /// VM is already created
    VmAlreadyCreated,

    /// VM is not running
    VmNotRunning,

    /// Cannot clone EventFd.
    EventFdClone(io::Error),

    /// Invalid VM state transition
    InvalidStateTransition(VmState, VmState),

    /// Error from CPU handling
    CpuManager(cpu::Error),

    /// Memory manager error
    MemoryManager(MemoryManagerError),

    /// No PCI support
    NoPciSupport,

    /// Eventfd write error
    EventfdError(std::io::Error),

    /// Cannot convert source URL from Path into &str
    RestoreSourceUrlPathToStr,

    /// Failed to validate config
    ConfigValidation(ValidationError),

    /// No more that one virtio-vsock device
    TooManyVsockDevices,
}
pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum VmState {
    Created,
    Running,
    Shutdown,
    Paused,
}

impl VmState {
    fn valid_transition(self, new_state: VmState) -> Result<()> {
        match self {
            VmState::Created => match new_state {
                VmState::Created | VmState::Shutdown | VmState::Paused => {
                    Err(Error::InvalidStateTransition(self, new_state))
                }
                VmState::Running => Ok(()),
            },

            VmState::Running => match new_state {
                VmState::Created | VmState::Running => {
                    Err(Error::InvalidStateTransition(self, new_state))
                }
                VmState::Paused | VmState::Shutdown => Ok(()),
            },

            VmState::Shutdown => match new_state {
                VmState::Paused | VmState::Created | VmState::Shutdown => {
                    Err(Error::InvalidStateTransition(self, new_state))
                }
                VmState::Running => Ok(()),
            },

            VmState::Paused => match new_state {
                VmState::Created | VmState::Paused => {
                    Err(Error::InvalidStateTransition(self, new_state))
                }
                VmState::Running | VmState::Shutdown => Ok(()),
            },
        }
    }
}

pub struct Vm {
    kernel: File,
    initramfs: Option<File>,
    threads: Vec<thread::JoinHandle<()>>,
    device_manager: Arc<Mutex<DeviceManager>>,
    config: Arc<Mutex<VmConfig>>,
    on_tty: bool,
    signals: Option<Signals>,
    state: RwLock<VmState>,
    cpu_manager: Arc<Mutex<cpu::CpuManager>>,
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl Vm {
    fn bhyve_new() -> Result<(VMMSystem, Arc<VirtualMachine>)> {
        let system = VMMSystem::new().map_err(Error::BhyveNew)?;
        system.create_vm("randomname").map_err(Error::VmCreate)?;
        let vm = VirtualMachine::new("randomname").map_err(Error::VmCreate)?;

        let fd = Arc::new(vm);

        Ok((system, fd))
    }

    fn new_from_memory_manager(
        config: Arc<Mutex<VmConfig>>,
        memory_manager: Arc<Mutex<MemoryManager>>,
        fd: Arc<VirtualMachine>,
        system: VMMSystem,
        exit_evt: EventFd,
        reset_evt: EventFd,
    ) -> Result<Self> {
        config
            .lock()
            .unwrap()
            .validate()
            .map_err(Error::ConfigValidation)?;

        let device_manager = DeviceManager::new(
            fd.clone(),
            config.clone(),
            memory_manager.clone(),
            &exit_evt,
            &reset_evt,
        )
        .map_err(Error::DeviceManager)?;

        let cpu_manager = cpu::CpuManager::new(
            &config.lock().unwrap().cpus.clone(),
            &device_manager,
            memory_manager.lock().unwrap().guest_memory(),
            &system,
            fd,
            reset_evt,
        )
        .map_err(Error::CpuManager)?;

        let on_tty = unsafe { libc::isatty(libc::STDIN_FILENO as i32) } != 0;
        let kernel = File::open(&config.lock().unwrap().kernel.as_ref().unwrap().path)
            .map_err(Error::KernelFile)?;

        let initramfs = config
            .lock()
            .unwrap()
            .initramfs
            .as_ref()
            .map(|i| File::open(&i.path))
            .transpose()
            .map_err(Error::InitramfsFile)?;

        Ok(Vm {
            kernel,
            initramfs,
            device_manager,
            config,
            on_tty,
            threads: Vec::with_capacity(1),
            signals: None,
            state: RwLock::new(VmState::Created),
            cpu_manager,
            memory_manager,
        })
    }

    pub fn new(
        config: Arc<Mutex<VmConfig>>,
        exit_evt: EventFd,
        reset_evt: EventFd,
    ) -> Result<Self> {
        let (system, fd) = Vm::bhyve_new()?;
        let memory_manager = MemoryManager::new(
            fd.clone(),
            &config.lock().unwrap().memory.clone(),
            None,
        )
        .map_err(Error::MemoryManager)?;

        let vm = Vm::new_from_memory_manager(
            config,
            memory_manager,
            fd,
            system,
            exit_evt,
            reset_evt,
        )?;

        // The device manager must create the devices from here as it is part
        // of the regular code path creating everything from scratch.
        vm.device_manager
            .lock()
            .unwrap()
            .create_devices()
            .map_err(Error::DeviceManager)?;

        Ok(vm)
    }

    fn load_initramfs(&mut self, guest_mem: &GuestMemoryMmap) -> Result<arch::InitramfsConfig> {
        let mut initramfs = self.initramfs.as_ref().unwrap();
        let size: usize = initramfs
            .seek(SeekFrom::End(0))
            .map_err(|_| Error::InitramfsLoad)?
            .try_into()
            .unwrap();
        initramfs
            .seek(SeekFrom::Start(0))
            .map_err(|_| Error::InitramfsLoad)?;

        let address =
            arch::initramfs_load_addr(guest_mem, size).map_err(|_| Error::InitramfsLoad)?;
        let address = GuestAddress(address);

        guest_mem
            .read_from(address, &mut initramfs, size)
            .map_err(|_| Error::InitramfsLoad)?;

        Ok(arch::InitramfsConfig { address, size })
    }

    fn load_kernel(&mut self) -> Result<EntryPoint> {
        let mut cmdline = Cmdline::new(arch::CMDLINE_MAX_SIZE);
        cmdline
            .insert_str(self.config.lock().unwrap().cmdline.args.clone())
            .map_err(Error::CmdLineInsertStr)?;
        for entry in self.device_manager.lock().unwrap().cmdline_additions() {
            cmdline.insert_str(entry).map_err(Error::CmdLineInsertStr)?;
        }

        let cmdline_cstring = CString::new(cmdline).map_err(Error::CmdLineCString)?;
        let guest_memory = self.memory_manager.lock().as_ref().unwrap().guest_memory();
        let mem = guest_memory.memory();
        let entry_addr = match linux_loader::loader::elf::Elf::load(
            mem.deref(),
            None,
            &mut self.kernel,
            Some(arch::layout::HIGH_RAM_START),
        ) {
            Ok(entry_addr) => entry_addr,
            Err(linux_loader::loader::Error::Elf(InvalidElfMagicNumber)) => {
                linux_loader::loader::bzimage::BzImage::load(
                    mem.deref(),
                    None,
                    &mut self.kernel,
                    Some(arch::layout::HIGH_RAM_START),
                )
                .map_err(Error::KernelLoad)?
            }
            Err(e) => {
                return Err(Error::KernelLoad(e));
            }
        };

        linux_loader::loader::load_cmdline(
            mem.deref(),
            arch::layout::CMDLINE_START,
            &cmdline_cstring,
        )
        .map_err(Error::LoadCmdLine)?;

        let initramfs_config = match self.initramfs {
            Some(_) => Some(self.load_initramfs(mem.deref())?),
            None => None,
        };

        let boot_vcpus = self.cpu_manager.lock().unwrap().boot_vcpus();
        let _max_vcpus = self.cpu_manager.lock().unwrap().max_vcpus();

        #[allow(unused_mut, unused_assignments)]
        let mut rsdp_addr: Option<GuestAddress> = None;

        #[cfg(feature = "acpi")]
        {
            rsdp_addr = Some(bhyve_oxidase::acpi::create_acpi_tables(
                mem.deref(),
                &self.device_manager,
                &self.cpu_manager,
                &self.memory_manager,
            ));
        }

        match entry_addr.setup_header {
            Some(hdr) => {
                arch::configure_system(
                    &mem,
                    arch::layout::CMDLINE_START,
                    cmdline_cstring.to_bytes().len() + 1,
                    &initramfs_config,
                    boot_vcpus,
                    Some(hdr),
                    rsdp_addr,
                    BootProtocol::LinuxBoot,
                )
                .map_err(Error::ConfigureSystem)?;

                let load_addr = entry_addr
                    .kernel_load
                    .raw_value()
                    .checked_add(KERNEL_64BIT_ENTRY_OFFSET)
                    .ok_or(Error::MemOverflow)?;

                Ok(EntryPoint {
                    entry_addr: GuestAddress(load_addr),
                    protocol: BootProtocol::LinuxBoot,
                })
            }
            None => {
                let entry_point_addr: GuestAddress;
                let boot_prot: BootProtocol;

                if let Some(pvh_entry_addr) = entry_addr.pvh_entry_addr {
                    // Use the PVH kernel entry point to boot the guest
                    entry_point_addr = pvh_entry_addr;
                    boot_prot = BootProtocol::PvhBoot;
                } else {
                    // Use the Linux 64-bit boot protocol
                    entry_point_addr = entry_addr.kernel_load;
                    boot_prot = BootProtocol::LinuxBoot;
                }

                arch::configure_system(
                    &mem,
                    arch::layout::CMDLINE_START,
                    cmdline_cstring.to_bytes().len() + 1,
                    &initramfs_config,
                    boot_vcpus,
                    None,
                    rsdp_addr,
                    boot_prot,
                )
                .map_err(Error::ConfigureSystem)?;

                Ok(EntryPoint {
                    entry_addr: entry_point_addr,
                    protocol: boot_prot,
                })
            }
        }
    }

    pub fn shutdown(&mut self) -> Result<()> {
        let current_state = self.get_state()?;
        let new_state = VmState::Shutdown;

        current_state.valid_transition(new_state)?;

        if self.on_tty {
            // Don't forget to set the terminal in canonical mode
            // before to exit.
            io::stdin()
                .lock()
                .set_canon_mode()
                .map_err(Error::SetTerminalCanon)?;
        }

        // Trigger the termination of the signal_handler thread
        if let Some(signals) = self.signals.take() {
            signals.close();
        }

        self.cpu_manager
            .lock()
            .unwrap()
            .shutdown()
            .map_err(Error::CpuManager)?;

        // Wait for all the threads to finish
        for thread in self.threads.drain(..) {
            thread.join().map_err(Error::ThreadCleanup)?
        }

        let mut state = self.state.try_write().map_err(|_| Error::PoisonedState)?;
        *state = new_state;

        Ok(())
    }

    pub fn add_device(&mut self, mut _device_cfg: DeviceConfig) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                self.device_manager
                    .lock()
                    .unwrap()
                    .add_device(&mut _device_cfg)
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by adding the new device. This is important to
                // ensure the device would be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();
                    if let Some(devices) = config.devices.as_mut() {
                        devices.push(_device_cfg);
                    } else {
                        config.devices = Some(vec![_device_cfg]);
                    }
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    pub fn remove_device(&mut self, _id: String) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                self.device_manager
                    .lock()
                    .unwrap()
                    .remove_device(_id.clone())
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by removing the device. This is important to
                // ensure the device would not be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();

                    // Remove if VFIO device
                    if let Some(devices) = config.devices.as_mut() {
                        devices.retain(|dev| dev.id.as_ref() != Some(&_id));
                    }

                    // Remove if disk device
                    if let Some(disks) = config.disks.as_mut() {
                        disks.retain(|dev| dev.id.as_ref() != Some(&_id));
                    }

                    // Remove if net device
                    if let Some(net) = config.net.as_mut() {
                        net.retain(|dev| dev.id.as_ref() != Some(&_id));
                    }

                    // Remove if pmem device
                    if let Some(pmem) = config.pmem.as_mut() {
                        pmem.retain(|dev| dev.id.as_ref() != Some(&_id));
                    }

                    // Remove if vsock device
                    if let Some(vsock) = config.vsock.as_ref() {
                        if vsock.id.as_ref() == Some(&_id) {
                            config.vsock = None;
                        }
                    }
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    pub fn add_disk(&mut self, mut _disk_cfg: DiskConfig) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                self.device_manager
                    .lock()
                    .unwrap()
                    .add_disk(&mut _disk_cfg)
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by adding the new device. This is important to
                // ensure the device would be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();
                    if let Some(disks) = config.disks.as_mut() {
                        disks.push(_disk_cfg);
                    } else {
                        config.disks = Some(vec![_disk_cfg]);
                    }
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    pub fn add_fs(&mut self, mut _fs_cfg: FsConfig) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                self.device_manager
                    .lock()
                    .unwrap()
                    .add_fs(&mut _fs_cfg)
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by adding the new device. This is important to
                // ensure the device would be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();
                    if let Some(fs_config) = config.fs.as_mut() {
                        fs_config.push(_fs_cfg);
                    } else {
                        config.fs = Some(vec![_fs_cfg]);
                    }
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    pub fn add_pmem(&mut self, mut _pmem_cfg: PmemConfig) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                self.device_manager
                    .lock()
                    .unwrap()
                    .add_pmem(&mut _pmem_cfg)
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by adding the new device. This is important to
                // ensure the device would be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();
                    if let Some(pmem) = config.pmem.as_mut() {
                        pmem.push(_pmem_cfg);
                    } else {
                        config.pmem = Some(vec![_pmem_cfg]);
                    }
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    pub fn add_net(&mut self, mut _net_cfg: NetConfig) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                self.device_manager
                    .lock()
                    .unwrap()
                    .add_net(&mut _net_cfg)
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by adding the new device. This is important to
                // ensure the device would be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();
                    if let Some(net) = config.net.as_mut() {
                        net.push(_net_cfg);
                    } else {
                        config.net = Some(vec![_net_cfg]);
                    }
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    pub fn add_vsock(&mut self, mut _vsock_cfg: VsockConfig) -> Result<()> {
        if cfg!(feature = "pci_support") {
            #[cfg(feature = "pci_support")]
            {
                if self.config.lock().unwrap().vsock.is_some() {
                    return Err(Error::TooManyVsockDevices);
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .add_vsock(&mut _vsock_cfg)
                    .map_err(Error::DeviceManager)?;

                // Update VmConfig by adding the new device. This is important to
                // ensure the device would be created in case of a reboot.
                {
                    let mut config = self.config.lock().unwrap();
                    config.vsock = Some(_vsock_cfg);
                }

                self.device_manager
                    .lock()
                    .unwrap()
                    .notify_hotplug(HotPlugNotificationFlags::PCI_DEVICES_CHANGED)
                    .map_err(Error::DeviceManager)?;
            }
            Ok(())
        } else {
            Err(Error::NoPciSupport)
        }
    }

    fn os_signal_handler(signals: Signals, console_input_clone: Arc<Console>, on_tty: bool) {
        for signal in signals.forever() {
            match signal {
                SIGWINCH => {
                    let (col, row) = get_win_size();
                    console_input_clone.update_console_size(col, row);
                }
                SIGTERM | SIGINT => {
                    if on_tty {
                        io::stdin()
                            .lock()
                            .set_canon_mode()
                            .expect("failed to restore terminal mode");
                    }
                    std::process::exit((signal != SIGTERM) as i32);
                }
                _ => (),
            }
        }
    }

    pub fn boot(&mut self) -> Result<()> {
        let current_state = self.get_state()?;

        let new_state = VmState::Running;
        current_state.valid_transition(new_state)?;

        let entry_addr = self.load_kernel()?;

        self.cpu_manager
            .lock()
            .unwrap()
            .start_boot_vcpus(entry_addr)
            .map_err(Error::CpuManager)?;

        if self
            .device_manager
            .lock()
            .unwrap()
            .console()
            .input_enabled()
        {
            let console = self.device_manager.lock().unwrap().console().clone();
            let signals = Signals::new(&[SIGWINCH, SIGINT, SIGTERM]);
            match signals {
                Ok(signals) => {
                    self.signals = Some(signals.clone());

                    let on_tty = self.on_tty;
                    self.threads.push(
                        thread::Builder::new()
                            .name("signal_handler".to_string())
                            .spawn(move || Vm::os_signal_handler(signals, console, on_tty))
                            .map_err(Error::SignalHandlerSpawn)?,
                    );
                }
                Err(e) => error!("Signal not found {}", e),
            }

            if self.on_tty {
                io::stdin()
                    .lock()
                    .set_raw_mode()
                    .map_err(Error::SetTerminalRaw)?;
            }
        }

        let mut state = self.state.try_write().map_err(|_| Error::PoisonedState)?;
        *state = new_state;

        Ok(())
    }

    pub fn handle_stdin(&self) -> Result<()> {
        let mut out = [0u8; 64];
        let count = io::stdin()
            .lock()
            .read_raw(&mut out)
            .map_err(Error::Console)?;

        if self
            .device_manager
            .lock()
            .unwrap()
            .console()
            .input_enabled()
        {
            self.device_manager
                .lock()
                .unwrap()
                .console()
                .queue_input_bytes(&out[..count])
                .map_err(Error::Console)?;
        }

        Ok(())
    }

    /// Gets a thread-safe reference counted pointer to the VM configuration.
    pub fn get_config(&self) -> Arc<Mutex<VmConfig>> {
        Arc::clone(&self.config)
    }

    /// Get the VM state. Returns an error if the state is poisoned.
    pub fn get_state(&self) -> Result<VmState> {
        self.state
            .try_read()
            .map_err(|_| Error::PoisonedState)
            .map(|state| *state)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test_vm_state_transitions(state: VmState) {
        match state {
            VmState::Created => {
                // Check the transitions from Created
                assert!(state.valid_transition(VmState::Created).is_err());
                assert!(state.valid_transition(VmState::Running).is_ok());
                assert!(state.valid_transition(VmState::Shutdown).is_err());
                assert!(state.valid_transition(VmState::Paused).is_err());
            }
            VmState::Running => {
                // Check the transitions from Running
                assert!(state.valid_transition(VmState::Created).is_err());
                assert!(state.valid_transition(VmState::Running).is_err());
                assert!(state.valid_transition(VmState::Shutdown).is_ok());
                assert!(state.valid_transition(VmState::Paused).is_ok());
            }
            VmState::Shutdown => {
                // Check the transitions from Shutdown
                assert!(state.valid_transition(VmState::Created).is_err());
                assert!(state.valid_transition(VmState::Running).is_ok());
                assert!(state.valid_transition(VmState::Shutdown).is_err());
                assert!(state.valid_transition(VmState::Paused).is_err());
            }
            VmState::Paused => {
                // Check the transitions from Paused
                assert!(state.valid_transition(VmState::Created).is_err());
                assert!(state.valid_transition(VmState::Running).is_ok());
                assert!(state.valid_transition(VmState::Shutdown).is_ok());
                assert!(state.valid_transition(VmState::Paused).is_err());
            }
        }
    }

    #[test]
    fn test_vm_created_transitions() {
        test_vm_state_transitions(VmState::Created);
    }

    #[test]
    fn test_vm_running_transitions() {
        test_vm_state_transitions(VmState::Running);
    }

    #[test]
    fn test_vm_shutdown_transitions() {
        test_vm_state_transitions(VmState::Shutdown);
    }

    #[test]
    fn test_vm_paused_transitions() {
        test_vm_state_transitions(VmState::Paused);
    }
}

#[allow(unused)]
pub fn test_vm() {
    // This example based on https://lwn.net/Articles/658511/
    let code = [
        0xba, 0xf8, 0x03, /* mov $0x3f8, %dx */
        0x00, 0xd8, /* add %bl, %al */
        0x04, b'0', /* add $'0', %al */
        0xee, /* out %al, (%dx) */
        0xb0, b'\n', /* mov $'\n', %al */
        0xee,  /* out %al, (%dx) */
        0xf4,  /* hlt */
    ];

    let vm_name = "testasm";
    let mem_size: usize = 20 * 1024 * 1024;
    let guest_addr: u64 = 0xfff0;

    let low_start_addr = GuestAddress(0);
    let no_offset = MemoryRegionAddress(0);

    let load_addr = GuestAddress(guest_addr);
    let mem = GuestMemoryMmap::from_ranges(&[(low_start_addr, mem_size)]).unwrap();

    let system = VMMSystem::new().expect("new Bhyve instance creation failed");
    system.create_vm(vm_name).expect("new VM fd creation failed");
    let vm_fd = VirtualMachine::new(vm_name).expect("new VM fd creation failed");

    // Lookup the start address for the memory region in host memory
    let region = mem.find_region(low_start_addr).unwrap();
    let host_addr = region.get_host_address(no_offset).unwrap();

    // Setup the guest physical memory in the guest
    vm_fd.setup_lowmem(host_addr as u64, mem_size).expect("failed to set guest memory");

    mem.write_slice(&code, load_addr)
        .expect("Writing code to memory failed");

    vm_fd.reinit().expect("failed to re-initialize VM");
    vm_fd.set_topology(1, 1, 1).expect("failed to set CPU topology");
    vm_fd.set_x2apic_state(0, true).expect("failed to enable x2APIC");
    vm_fd.set_capability(0, vm_cap_type::VM_CAP_UNRESTRICTED_GUEST, 1).expect("unrestricted guest capability not available");
    vm_fd.set_capability(0, vm_cap_type::VM_CAP_HALT_EXIT, 1).expect("exit on halt guest capability not available");

    let lomem = (mem_size - (16 * 1024 * 1024)) / (64 * 1024);
    vm_fd.rtc_write(0x34, lomem as u8).expect("failed to set RTC memory size");
    vm_fd.rtc_write(0x35, (lomem >> 8) as u8).expect("failed to set RTC memory size");

    // Setup registers
    vm_fd.vcpu_reset(0).expect("failed to set initial state of registers");

    let (_base, limit, access) = vm_fd.get_desc(0, vm_reg_name::VM_REG_GUEST_CS).expect("failed to get CS desc");
    vm_fd.set_desc(0, vm_reg_name::VM_REG_GUEST_CS, guest_addr, limit, access).expect("failed to set CS desc");

    vm_fd.set_register(0, vm_reg_name::VM_REG_GUEST_RAX, 2).expect("failed to set RAX register");
    vm_fd.set_register(0, vm_reg_name::VM_REG_GUEST_RBX, 3).expect("failed to set RBX register");
    vm_fd.set_register(0, vm_reg_name::VM_REG_GUEST_RFLAGS, 2).expect("failed to set RFLAGS register");

    match vm_fd.activate_vcpu(0) {
        Ok(_) => println!("Activated CPU 0 for VM at /dev/vmm/{}", vm_name),
        Err(e) => println!("Failed to activate CPU 0 for VM at /dev/vmm/{}, with error: {}", vm_name, e),
    };


    loop {
        match vm_fd.run(0).expect("run failed") {
            VmExit::IoIn(addr, bytes) => {
                println!(
                    "IO in -- addr: {:#x} bytes {:?}",
                    addr,
                    bytes
                );
            }
            VmExit::IoOut(addr, bytes, value) => {
                let mut data: [u8; 4] = value.to_le_bytes();
                println!(
                    "IO out -- addr: {:#x} bytes {:?} value [{:?}]",
                    addr,
                    bytes,
                    str::from_utf8(&data).unwrap()
                );
            }
            VmExit::Vmx(s, r, q, t, e) => {
                println!("exit for Vmx, source={}, reason={}, qualification={:b}, inst type={}, inst error={}", s, r, q, t, e);
                if r == 2 {
                    println!("Exit reason is triple fault");
                    break;
                }
            }
            VmExit::Bogus => {
                println!("exit for Bogus");
                break;
            }
            VmExit::Halt => {
                println!("exit for Halt");
                break;
            }
            VmExit::Suspended => {
                println!("exit for Suspended");
                break;
            }
            reason => println!("Unhandled exit reason {:?}", reason)
        }
    }
}
