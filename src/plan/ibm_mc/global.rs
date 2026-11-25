use super::gc_work::IBMMCWorkContext;
use super::gc_work::{ForwardingProcessEdges, MarkingProcessEdges};
use crate::plan::compressor::gc_work::GenerateWork;
use crate::plan::compressor::process_edges::PlanRemember;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::global::{BasePlan, CommonPlan};
use crate::plan::ibm_mc::mutator::ALLOCATOR_MAPPING;
use crate::plan::plan_constraints::MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN;
use crate::plan::{AllocationSemantics, Plan, PlanConstraints};
use crate::policy::ibm_mc::IBMMCSpace;
use crate::policy::space::Space;
use crate::scheduler::gc_work::*;
use crate::scheduler::{GCWorkScheduler, GCWorker, WorkBucketStage};
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::heap::gc_trigger::SpaceStats;
#[allow(unused_imports)]
use crate::util::heap::VMRequest;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::opaque_pointer::*;
use crate::util::remset::RemSet;
use crate::util::ObjectReference;
use crate::vm::VMBinding;
use enum_map::EnumMap;
use mmtk_macros::{HasSpaces, PlanTraceObject};

/// [`IBMMC`] implements a stop-the-world, single-threaded implementation of
/// the IBM Mark-Compact algorithm, as described in Abuaiadh et al.,
/// [An Efficient Parallel Heap Compaction Algorithm](https://dl.acm.org/doi/10.1145/1028976.1028995).
#[derive(HasSpaces, PlanTraceObject)]
pub struct IBMMC<VM: VMBinding> {
    #[parent]
    pub common: CommonPlan<VM>,
    #[space]
    pub ibm_mc_space: IBMMCSpace<VM>,
    remset: RemSet<VM>,
}

/// The plan constraints for the IBM MC plan.
pub const IBM_MC_CONSTRAINTS: PlanConstraints = PlanConstraints {
    max_non_los_default_alloc_bytes: MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
    moves_objects: true,
    needs_forward_after_liveness: true,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for IBMMC<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &IBM_MC_CONSTRAINTS
    }

    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        self.base().collection_required(self, space_full)
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.common
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.common.base
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        self.common.prepare(tls, true);
        self.ibm_mc_space.prepare();
    }

    fn release(&mut self, tls: VMWorkerThread) {
        self.common.release(tls, true);
        self.ibm_mc_space.release();
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        self.common.end_of_gc(tls);
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        // Stop & scan mutators (mutator scanning can happen before STW)
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<IBMMCWorkContext<VM>>::new());

        // Prepare global/collectors/mutators
        scheduler.work_buckets[WorkBucketStage::Prepare]
            .add(Prepare::<IBMMCWorkContext<VM>>::new(self));

        // Phase 1 (Moving Phase): compact objects and compute offset vector + alloc bitmap.
        scheduler.work_buckets[WorkBucketStage::CalculateForwarding]
            .add_boxed(self.ibm_mc_space.move_objects_task());

        // Update remembered external/root slots after Phase 1 has populated forwarding metadata.
        scheduler.work_buckets[WorkBucketStage::SecondRoots].add(GenerateWork::new(|| {
            self.ibm_mc_space
                .add_remset_tasks(&self.remset, WorkBucketStage::SecondRoots)
        }));

        // Phase 2 (Fixup Phase): scan relocated live objects via alloc bitmap and update references.
        scheduler.work_buckets[WorkBucketStage::Compact]
            .add_boxed(self.ibm_mc_space.fixup_task());

        // Release global/collectors/mutators
        scheduler.work_buckets[WorkBucketStage::Release]
            .add(Release::<IBMMCWorkContext<VM>>::new(self));

        // Reference processing
        if !*self.base().options.no_reference_types {
            use crate::util::reference_processor::{
                PhantomRefProcessing, SoftRefProcessing, WeakRefProcessing,
            };
            scheduler.work_buckets[WorkBucketStage::SoftRefClosure]
                .add(SoftRefProcessing::<MarkingProcessEdges<VM>>::new());
            scheduler.work_buckets[WorkBucketStage::WeakRefClosure]
                .add(WeakRefProcessing::<VM>::new());
            scheduler.work_buckets[WorkBucketStage::PhantomRefClosure]
                .add(PhantomRefProcessing::<VM>::new());

            use crate::util::reference_processor::RefForwarding;
            scheduler.work_buckets[WorkBucketStage::RefForwarding]
                .add(RefForwarding::<ForwardingProcessEdges<VM>>::new());

            use crate::util::reference_processor::RefEnqueue;
            scheduler.work_buckets[WorkBucketStage::Release].add(RefEnqueue::<VM>::new());
        }

        // Finalization
        if !*self.base().options.no_finalizer {
            use crate::util::finalizable_processor::{Finalization, ForwardFinalization};
            scheduler.work_buckets[WorkBucketStage::FinalRefClosure]
                .add(Finalization::<MarkingProcessEdges<VM>>::new());
            scheduler.work_buckets[WorkBucketStage::FinalizableForwarding]
                .add(ForwardFinalization::<ForwardingProcessEdges<VM>>::new());
        }

        // VM-specific weak ref processing
        scheduler.work_buckets[WorkBucketStage::VMRefClosure]
            .set_sentinel(Box::new(VMProcessWeakRefs::<MarkingProcessEdges<VM>>::new()));

        // VM-specific weak ref forwarding
        scheduler.work_buckets[WorkBucketStage::VMRefForwarding]
            .add(VMForwardWeakRefs::<ForwardingProcessEdges<VM>>::new());

        // VM-specific work after forwarding
        scheduler.work_buckets[WorkBucketStage::Release].add(VMPostForwarding::<VM>::default());

        // Analysis GC work
        #[cfg(feature = "analysis")]
        {
            use crate::util::analysis::GcHookWork;
            scheduler.work_buckets[WorkBucketStage::Unconstrained].add(GcHookWork);
        }
        #[cfg(feature = "sanity")]
        scheduler.work_buckets[WorkBucketStage::Final]
            .add(crate::util::sanity::sanity_checker::ScheduleSanityGC::<Self>::new(self));
    }

    fn current_gc_may_move_object(&self) -> bool {
        true
    }

    fn get_used_pages(&self) -> usize {
        self.ibm_mc_space.reserved_pages() + self.common.get_used_pages()
    }
}

impl<VM: VMBinding> IBMMC<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let scheduler = args.scheduler.clone();

        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &IBM_MC_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&[]),
        };

        let res = IBMMC {
            ibm_mc_space: IBMMCSpace::new(plan_args.get_normal_space_args(
                "ibm_mc_space",
                true,
                false,
                VMRequest::discontiguous(),
            )),
            common: CommonPlan::new(plan_args),
            remset: RemSet::new(scheduler.num_workers()),
        };

        res.verify_side_metadata_sanity();

        res
    }
}

impl<VM: VMBinding> PlanRemember<VM> for IBMMC<VM> {
    fn record(&self, source: VM::VMSlot, target: ObjectReference, worker: &GCWorker<VM>) {
        self.remset.record(source, target, worker);
    }
}
