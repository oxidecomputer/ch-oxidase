// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//

use devices::ioapic;
use bhyve_api::vm::VirtualMachine;
use std::collections::HashMap;
use std::io;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use vm_allocator::SystemAllocator;
use vm_device::interrupt::{
    InterruptIndex, InterruptManager, InterruptSourceConfig, InterruptSourceGroup,
    LegacyIrqGroupConfig, MsiIrqGroupConfig,
};
use vmm_sys_util::eventfd::EventFd;

/// Reuse std::io::Result to simplify interoperability among crates.
pub type Result<T> = std::io::Result<T>;

pub struct InterruptRoute {
    pub gsi: u32,
    registered: AtomicBool,
}

impl InterruptRoute {
    pub fn new(allocator: &mut SystemAllocator) -> Result<Self> {
        let gsi = allocator
            .allocate_gsi()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Failed allocating new GSI"))?;

        Ok(InterruptRoute {
            gsi,
            registered: AtomicBool::new(false),
        })
    }

    pub fn enable(&self) -> Result<()> {
        if !self.registered.load(Ordering::SeqCst) {
            // Update internals to track the interrupt route as "registered".
            self.registered.store(true, Ordering::SeqCst);
        }

        Ok(())
    }

    pub fn disable(&self) -> Result<()> {
        if self.registered.load(Ordering::SeqCst) {
            // Update internals to track the irq_fd as "unregistered".
            self.registered.store(false, Ordering::SeqCst);
        }

        Ok(())
    }
}


#[derive(Default)]
pub struct MsiRoutingEntry {
    gsi: u32,
    flags: u32,
    address_lo: u32,
    address_hi: u32,
    data: u32,
    devid: u32,
    masked: bool,
}

pub struct MsiInterruptGroup {
    vm_fd: Arc<VirtualMachine>,
    gsi_msi_routes: Arc<Mutex<HashMap<u32, MsiRoutingEntry>>>,
    irq_routes: HashMap<InterruptIndex, InterruptRoute>,
}

impl MsiInterruptGroup {
    fn new(
        vm_fd: Arc<VirtualMachine>,
        gsi_msi_routes: Arc<Mutex<HashMap<u32, MsiRoutingEntry>>>,
        irq_routes: HashMap<InterruptIndex, InterruptRoute>,
    ) -> Self {
        MsiInterruptGroup {
            vm_fd,
            gsi_msi_routes,
            irq_routes,
        }
    }

    fn mask_routing_entry(&self, index: InterruptIndex, mask: bool) -> Result<()> {
        if let Some(route) = self.irq_routes.get(&index) {
            let mut gsi_msi_routes = self.gsi_msi_routes.lock().unwrap();
            if let Some(routing_entry) = gsi_msi_routes.get_mut(&route.gsi) {
                routing_entry.masked = mask;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("mask: No existing route for interrupt index {}", index),
                ));
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("mask: Invalid interrupt index {}", index),
            ));
        }

        Ok(())
    }
}

impl InterruptSourceGroup for MsiInterruptGroup {
    fn enable(&self) -> Result<()> {
        for (_, route) in self.irq_routes.iter() {
            route.enable()?;
        }

        Ok(())
    }

    fn disable(&self) -> Result<()> {
        for (_, route) in self.irq_routes.iter() {
            route.disable()?;
        }

        Ok(())
    }

    fn trigger(&self, index: InterruptIndex) -> Result<()> {
        if let Some(route) = self.irq_routes.get(&index) {
            let gsi_msi_routes = self.gsi_msi_routes.lock().unwrap();
            if let Some(routing_entry) = gsi_msi_routes.get(&route.gsi) {
                // Inject an MSI interrupt into the guest
                match self.vm_fd.lapic_msi(routing_entry.address_lo as u64, routing_entry.data as u64) {
                    Ok(_) => return Ok(()),
                    Err(e) => return Err(e.into()),
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("trigger: Invalid interrupt index {}", index),
        ))
    }

    fn update(&self, index: InterruptIndex, config: InterruptSourceConfig) -> Result<()> {
        if let Some(route) = self.irq_routes.get(&index) {
            if let InterruptSourceConfig::MsiIrq(cfg) = &config {
                let routing_entry = MsiRoutingEntry {
                    gsi: route.gsi,
                    address_lo: cfg.low_addr,
                    address_hi: cfg.high_addr,
                    data: cfg.data,
                    masked: false,
                    ..Default::default()
                };

                self.gsi_msi_routes
                    .lock()
                    .unwrap()
                    .insert(route.gsi, routing_entry);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "Interrupt config type not supported",
                ));
            }
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("update: Invalid interrupt index {}", index),
        ))
    }

    fn mask(&self, index: InterruptIndex) -> Result<()> {
        self.mask_routing_entry(index, true)?;

        if let Some(route) = self.irq_routes.get(&index) {
            return route.disable();
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("mask: Invalid interrupt index {}", index),
        ))
    }

    fn unmask(&self, index: InterruptIndex) -> Result<()> {
        self.mask_routing_entry(index, false)?;

        if let Some(route) = self.irq_routes.get(&index) {
            return route.enable();
        }

        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("unmask: Invalid interrupt index {}", index),
        ))
    }
}

pub struct LegacyUserspaceInterruptGroup {
    ioapic: Arc<Mutex<ioapic::Ioapic>>,
    irq: u32,
}

impl LegacyUserspaceInterruptGroup {
    fn new(ioapic: Arc<Mutex<ioapic::Ioapic>>, irq: u32) -> Self {
        LegacyUserspaceInterruptGroup { ioapic, irq }
    }
}

impl InterruptSourceGroup for LegacyUserspaceInterruptGroup {
    fn trigger(&self, _index: InterruptIndex) -> Result<()> {
        self.ioapic
            .lock()
            .unwrap()
            .service_irq(self.irq as usize)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("failed to inject IRQ #{}: {:?}", self.irq, e),
                )
            })
    }

    fn update(&self, _index: InterruptIndex, _config: InterruptSourceConfig) -> Result<()> {
        Ok(())
    }
}

pub struct BhyveLegacyUserspaceInterruptManager {
    ioapic: Arc<Mutex<ioapic::Ioapic>>,
}

pub struct BhyveMsiInterruptManager {
    allocator: Arc<Mutex<SystemAllocator>>,
    vm_fd: Arc<VirtualMachine>,
    gsi_msi_routes: Arc<Mutex<HashMap<u32, MsiRoutingEntry>>>,
}

impl BhyveLegacyUserspaceInterruptManager {
    pub fn new(ioapic: Arc<Mutex<ioapic::Ioapic>>) -> Self {
        BhyveLegacyUserspaceInterruptManager { ioapic }
    }
}

impl BhyveMsiInterruptManager {
    pub fn new(
        allocator: Arc<Mutex<SystemAllocator>>,
        vm_fd: Arc<VirtualMachine>,
        gsi_msi_routes: Arc<Mutex<HashMap<u32, MsiRoutingEntry>>>,
    ) -> Self {
        BhyveMsiInterruptManager {
            allocator,
            vm_fd,
            gsi_msi_routes,
        }
    }
}

impl InterruptManager for BhyveLegacyUserspaceInterruptManager {
    type GroupConfig = LegacyIrqGroupConfig;

    fn create_group(
        &self,
        config: Self::GroupConfig,
    ) -> Result<Arc<Box<dyn InterruptSourceGroup>>> {
        Ok(Arc::new(Box::new(LegacyUserspaceInterruptGroup::new(
            self.ioapic.clone(),
            config.irq as u32,
        ))))
    }

    fn destroy_group(&self, _group: Arc<Box<dyn InterruptSourceGroup>>) -> Result<()> {
        Ok(())
    }
}

impl InterruptManager for BhyveMsiInterruptManager {
    type GroupConfig = MsiIrqGroupConfig;

    fn create_group(
        &self,
        config: Self::GroupConfig,
    ) -> Result<Arc<Box<dyn InterruptSourceGroup>>> {
        let mut allocator = self.allocator.lock().unwrap();
        let mut irq_routes: HashMap<InterruptIndex, InterruptRoute> =
            HashMap::with_capacity(config.count as usize);
        for i in config.base..config.base + config.count {
            irq_routes.insert(i, InterruptRoute::new(&mut allocator)?);
        }

        Ok(Arc::new(Box::new(MsiInterruptGroup::new(
            self.vm_fd.clone(),
            self.gsi_msi_routes.clone(),
            irq_routes,
        ))))
    }

    fn destroy_group(&self, _group: Arc<Box<dyn InterruptSourceGroup>>) -> Result<()> {
        Ok(())
    }
}
