use super::global::IBMMC;
use crate::plan::compressor::process_edges::{PlanProcessEdgesRemset, RemsetCondition};
use crate::policy::ibm_mc::{TRACE_KIND_FORWARD, TRACE_KIND_MARK};
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::VMBinding;
use std::marker::PhantomData;

/// Condition for recording slots pointing into `IBMMCSpace` from outside `IBMMCSpace`.
pub struct IBMMCCondition<VM: VMBinding> {
    _p: PhantomData<VM>,
}

impl<VM: VMBinding> RemsetCondition<IBMMC<VM>, VM> for IBMMCCondition<VM> {
    fn relevant(plan: &IBMMC<VM>, source: VM::VMSlot, target: ObjectReference) -> bool {
        !plan.ibm_mc_space.address_in_space(source.as_address())
            && plan.ibm_mc_space.in_space(target)
    }
}

/// Marking trace
pub type MarkingProcessEdges<VM> =
    PlanProcessEdgesRemset<VM, IBMMC<VM>, IBMMCCondition<VM>, TRACE_KIND_MARK>;

pub struct IBMMCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for IBMMCWorkContext<VM> {
    type VM = VM;
    type PlanType = IBMMC<VM>;
    type DefaultProcessEdges = MarkingProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Forwarding trace (used for weak references / finalizers after the Moving phase)
pub type ForwardingProcessEdges<VM> = PlanProcessEdges<VM, IBMMC<VM>, TRACE_KIND_FORWARD>;
