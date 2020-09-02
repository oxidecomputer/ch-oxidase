# Illumos port of Cloud Hypervisor

This project is an experimental port of [Cloud
Hypervisor](https://github.com/cloud-hypervisor/cloud-hypervisor) to run on
Illumos and the Bhyve kernel space, instead of Linux and KVM. It is not a
mergable fork of Cloud Hypervisor, it's a work-in-progress first-pass bottom-up
translation of the modules and binaries that can be ported, to explore how
extensive and disruptive the changes to the source code will be. So far, it
seems likely that this work can be converted into a clean `hypervisor` trait
implementation and `arch` submodule within Cloud Hypervisor.

The Bhyve ioctl interface is wrapped in a separate Rust crate
[bhyve-api](https://github.com/oxidecomputer/bhyve-api), which is roughly
equivalent to [rust-vmm/kvm-ioctls](https://github.com/rust-vmm/kvm-ioctls), and
should probably be renamed to `bhyve-ioctls` for consistency.
