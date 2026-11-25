use super::global::ParallelIBMMC;
use crate::plan::compressor::process_edges::{PlanProcessEdgesRemset, RemsetCondition};
use crate::policy::parallel_ibm_mc::{TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::VMBinding;
use std::marker::PhantomData;

/// Condition for recording slots pointing into `ParallelIBMMCSpace` from outside `ParallelIBMMCSpace`.
pub struct ParallelIBMMCCondition<VM: VMBinding> {
    _p: PhantomData<VM>,
}

impl<VM: VMBinding> RemsetCondition<ParallelIBMMC<VM>, VM> for ParallelIBMMCCondition<VM> {
    fn relevant(plan: &ParallelIBMMC<VM>, source: VM::VMSlot, target: ObjectReference) -> bool {
        !plan
            .parallel_ibm_mc_space
            .address_in_space(source.as_address())
            && plan.parallel_ibm_mc_space.in_space(target)
    }
}

/// Marking trace
pub type MarkingProcessEdges<VM> =
    PlanProcessEdgesRemset<VM, ParallelIBMMC<VM>, ParallelIBMMCCondition<VM>, TRACE_KIND_MARK>;

pub struct ParallelIBMMCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for ParallelIBMMCWorkContext<VM> {
    type VM = VM;
    type PlanType = ParallelIBMMC<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Forwarding trace (used for weak references / finalizers after the Moving phase)
pub type ForwardingProcessEdges<VM> =
    PlanProcessEdges<VM, ParallelIBMMC<VM>, TRACE_KIND_FORWARD>;
