use super::forwarding;
use crate::plan::VectorObjectQueue;
use crate::policy::gc_work::{TraceKind, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::sft::{GCWorkerMutRef, SFT};
use crate::policy::space::{CommonSpace, Space};
use crate::scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage};
use crate::util::alloc::allocator::align_allocation_no_fill;
use crate::util::copy::CopySemantics;
use crate::util::heap::regionpageresource::AllocatedRegion;
use crate::util::heap::{PageResource, RegionPageResource};
use crate::util::linear_scan::Region;
use crate::util::metadata::extract_side_metadata;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::ObjectEnumerator;
use crate::util::{Address, ObjectReference};
use crate::vm::slot::Slot;
use crate::MMTK;
use crate::{vm::*, ObjectQueue};
use atomic::Ordering;
use std::sync::Arc;

pub(crate) const TRACE_KIND_MARK: TraceKind = 0;
pub(crate) const TRACE_KIND_FORWARD: TraceKind = 1;

/// [`ParallelIBMMCSpace`] is a stop-the-world, parallel implementation of
/// the IBM Mark-Compact algorithm described in Abuaiadh et al.,
/// [An Efficient Parallel Heap Compaction Algorithm](https://dl.acm.org/doi/10.1145/1028976.1028995).
pub struct ParallelIBMMCSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: RegionPageResource<VM, forwarding::Area>,
    forwarding: forwarding::ForwardingMetadata<VM>,
    scheduler: Arc<GCWorkScheduler<VM>>,
}

impl<VM: VMBinding> SFT for ParallelIBMMCSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        if self.forwarding.has_calculated_forwarding_addresses() {
            Some(self.forward::<false>(object, false))
        } else {
            None
        }
    }

    fn is_live(&self, object: ObjectReference) -> bool {
        Self::is_marked(object)
    }

    #[cfg(feature = "object_pinning")]
    fn pin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of ParallelIBMMCSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, _object: ObjectReference) -> bool {
        panic!("Cannot pin/unpin objects of ParallelIBMMCSpace.")
    }

    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, _object: ObjectReference) -> bool {
        false
    }

    fn is_movable(&self) -> bool {
        true
    }

    fn initialize_object_metadata(&self, _object: ObjectReference) {
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(_object);
    }

    #[cfg(feature = "sanity")]
    fn is_sane(&self) -> bool {
        true
    }

    #[cfg(feature = "is_mmtk_object")]
    fn is_mmtk_object(&self, addr: Address) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::is_vo_bit_set_for_addr(addr)
    }

    #[cfg(feature = "is_mmtk_object")]
    fn find_object_from_internal_pointer(
        &self,
        ptr: Address,
        max_search_bytes: usize,
    ) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::find_object_from_internal_pointer::<VM>(
            ptr,
            max_search_bytes,
        )
    }

    fn sft_trace_object(
        &self,
        _queue: &mut VectorObjectQueue,
        _object: ObjectReference,
        _worker: GCWorkerMutRef,
    ) -> ObjectReference {
        panic!("sft_trace_object() cannot be used with ParallelIBMMCSpace")
    }

    fn debug_print_object_info(&self, object: ObjectReference) {
        println!(
            "marked = {}",
            ParallelIBMMCSpace::<VM>::is_marked(object)
        );
        println!("forwarding = {:?}", self.get_forwarded_object(object));
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for ParallelIBMMCSpace<VM> {
    fn as_space(&self) -> &dyn Space<VM> {
        self
    }

    fn as_sft(&self) -> &(dyn SFT + Sync + 'static) {
        self
    }

    fn get_page_resource(&self) -> &dyn PageResource<VM> {
        &self.pr
    }

    fn maybe_get_page_resource_mut(&mut self) -> Option<&mut dyn PageResource<VM>> {
        Some(&mut self.pr)
    }

    fn common(&self) -> &CommonSpace<VM> {
        &self.common
    }

    fn initialize_sft(&self, sft_map: &mut dyn crate::policy::sft_map::SFTMap) {
        self.common().initialize_sft(self.as_sft(), sft_map)
    }

    fn release_multiple_pages(&mut self, _start: Address) {
        panic!("parallelibmmcspace only releases pages enmasse")
    }

    fn enumerate_objects(&self, enumerator: &mut dyn ObjectEnumerator) {
        self.pr.enumerate(enumerator);
    }

    fn clear_side_log_bits(&self) {
        unimplemented!()
    }

    fn set_side_log_bits(&self) {
        unimplemented!()
    }
}

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for ParallelIBMMCSpace<VM> {
    fn trace_object<Q: ObjectQueue, const KIND: crate::policy::gc_work::TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        _copy: Option<CopySemantics>,
        _worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        debug_assert!(
            KIND != TRACE_KIND_TRANSITIVE_PIN,
            "ParallelIBMMCSpace does not support transitive pin trace."
        );
        if KIND == TRACE_KIND_MARK {
            self.trace_mark_object(queue, object)
        } else if KIND == TRACE_KIND_FORWARD {
            self.forward::<false>(object, true)
        } else {
            unreachable!()
        }
    }

    fn may_move_objects<const KIND: crate::policy::gc_work::TraceKind>() -> bool {
        if KIND == TRACE_KIND_MARK {
            false
        } else if KIND == TRACE_KIND_FORWARD {
            true
        } else {
            unreachable!()
        }
    }
}

impl<VM: VMBinding> ParallelIBMMCSpace<VM> {
    pub fn new(args: crate::policy::space::PlanCreateSpaceArgs<VM>) -> Self {
        let vm_map = args.vm_map;
        assert!(
            VM::VMObjectModel::UNIFIED_OBJECT_REFERENCE_ADDRESS,
            "ParallelIBMMCSpace requires a unified object reference address model"
        );
        let local_specs = extract_side_metadata(&[
            MetadataSpec::OnSide(forwarding::MARK_SPEC),
            MetadataSpec::OnSide(forwarding::ALLOC_SPEC),
            MetadataSpec::OnSide(forwarding::OFFSET_VECTOR_SPEC),
        ]);
        let is_discontiguous = args.vmrequest.is_discontiguous();
        let scheduler = args.scheduler.clone();
        let common = CommonSpace::new(args.into_policy_args(true, false, local_specs));
        ParallelIBMMCSpace {
            pr: if is_discontiguous {
                RegionPageResource::new_discontiguous(vm_map)
            } else {
                RegionPageResource::new_contiguous(common.start, common.extent, vm_map)
            },
            forwarding: forwarding::ForwardingMetadata::new(),
            common,
            scheduler,
        }
    }

    pub fn prepare(&self) {
        // Zero both MARK_SPEC and ALLOC_SPEC across the full span of each allocated area,
        // because cross-area compaction may move objects beyond a lower area's pre-GC cursor.
        self.pr
            .enumerate_regions(&mut |r: &AllocatedRegion<forwarding::Area>| {
                let start = r.region.start();
                let len = r.region.end() - start;
                forwarding::MARK_SPEC.bzero_metadata(start, len);
                forwarding::ALLOC_SPEC.bzero_metadata(start, len);
            });
    }

    pub fn release(&self) {
        self.forwarding.release();
    }

    pub fn trace_mark_object<Q: ObjectQueue>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
    ) -> ObjectReference {
        #[cfg(feature = "vo_bit")]
        debug_assert!(
            crate::util::metadata::vo_bit::is_vo_bit_set(object),
            "{:x}: VO bit not set",
            object
        );
        if ParallelIBMMCSpace::<VM>::test_and_mark(object) {
            queue.enqueue(object);
        }
        object
    }

    #[inline(always)]
    pub fn test_and_mark(object: ObjectReference) -> bool {
        let addr = object.to_raw_address();
        // Fast TTAS check avoids write invalidation on already-marked objects.
        if forwarding::MARK_SPEC.load_atomic::<u8>(addr, Ordering::Relaxed) != 0 {
            return false;
        }
        forwarding::MARK_SPEC.fetch_or_atomic::<u8>(addr, 1, Ordering::Relaxed) == 0
    }

    pub fn is_marked(object: ObjectReference) -> bool {
        forwarding::MARK_SPEC.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst) == 1
    }

    #[inline(always)]
    pub fn forward<const CAN_BMI2: bool>(
        &self,
        object: ObjectReference,
        _vo_bit_valid: bool,
    ) -> ObjectReference {
        if !self.in_space(object) {
            return object;
        }
        let addr = self
            .forwarding
            .forward::<CAN_BMI2>(object.to_raw_address());
        let ret = ObjectReference::from_raw_address(addr).unwrap();
        #[cfg(feature = "vo_bit")]
        if _vo_bit_valid {
            debug_assert!(
                crate::util::metadata::vo_bit::is_vo_bit_set(ret),
                "{:x}: VO bit not set",
                ret
            );
        }
        ret
    }

    pub fn add_remset_tasks(
        &'static self,
        remset: &crate::util::remset::RemSet<VM>,
        stage: WorkBucketStage,
    ) {
        fn inner<VM: VMBinding, const CAN_BMI2: bool>(
            this: &'static ParallelIBMMCSpace<VM>,
            remset: &crate::util::remset::RemSet<VM>,
            stage: WorkBucketStage,
        ) {
            let mut packets = vec![];
            remset.flush_all(&mut |entries| {
                let slots = entries.iter().map(|e| e.decode().0).collect();
                packets.push(Box::new(UpdateSlots::<VM, CAN_BMI2>::new(this, slots))
                    as Box<dyn GCWork<VM>>);
            });
            this.scheduler.work_buckets[stage].bulk_add(packets);
        }
        if self.forwarding.supports_bmi2() {
            inner::<VM, true>(self, remset, stage);
        } else {
            inner::<VM, false>(self, remset, stage);
        }
    }

    /// Spawn Phase 1 (`ParallelMove`) work packets across all workers.
    pub fn add_move_tasks(&'static self) {
        let mut regions = Vec::new();
        self.pr.enumerate_regions(&mut |r| {
            let cursor = r.cursor();
            #[cfg(feature = "vo_bit")]
            {
                let start = r.region.start();
                if cursor > start {
                    #[cfg(debug_assertions)]
                    self.forwarding
                        .scan_marked_objects(start, cursor, &mut |object: ObjectReference| {
                            debug_assert!(
                                crate::util::metadata::vo_bit::is_vo_bit_set(object),
                                "{:x}: VO bit not set",
                                object
                            );
                        });
                    crate::util::metadata::vo_bit::bzero_vo_bit(start, cursor - start);
                }
            }
            regions.push((r.region, cursor));
        });

        let num_regions = regions.len();
        self.forwarding.init_move_state(regions);

        if num_regions == 0 {
            return;
        }
        let num_packets = self.scheduler.num_workers().min(num_regions);
        let packets: Vec<Box<dyn GCWork<VM>>> = (0..num_packets)
            .map(|_| Box::new(ParallelMove::<VM>::new(self)) as Box<dyn GCWork<VM>>)
            .collect();
        self.scheduler.work_buckets[WorkBucketStage::CalculateForwarding].bulk_add(packets);
    }

    /// Execute Phase 1 (`Parallel_Move`) on the current GC worker.
    pub fn parallel_move(&self) {
        self.forwarding.run_parallel_move(
            &mut |obj, to| {
                let align = VM::VMObjectModel::get_align_when_copied(obj);
                let offset = VM::VMObjectModel::get_align_offset_when_copied(obj);
                let new_addr = align_allocation_no_fill::<VM>(to, align, offset);
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                new_addr + copied_size
            },
            &mut |obj, to| {
                let align = VM::VMObjectModel::get_align_when_copied(obj);
                let offset = VM::VMObjectModel::get_align_offset_when_copied(obj);
                let new_addr = align_allocation_no_fill::<VM>(to, align, offset);
                let new_obj = VM::VMObjectModel::get_reference_when_copied_to(obj, new_addr);
                debug_assert_eq!(new_obj.to_raw_address(), new_addr);
                let copied_size = VM::VMObjectModel::get_size_when_copied(obj);
                debug_assert_eq!(copied_size, VM::VMObjectModel::get_current_size(obj));

                if new_obj != obj {
                    trace!("copy from {} to {}", obj, new_obj);
                    let end_of_new_object =
                        VM::VMObjectModel::copy_to(obj, new_obj, Address::ZERO);
                    debug_assert_eq!(end_of_new_object, new_addr + copied_size);
                }

                #[cfg(feature = "vo_bit")]
                vo_bit::set_vo_bit(new_obj);

                (new_obj, new_addr + copied_size)
            },
        );
    }

    /// Finalize Phase 1 after all `ParallelMove` workers complete.
    pub fn after_move(&self) {
        let final_tops = self.forwarding.finish_move();
        self.pr.reset_cursors(&final_tops);
        self.pr.reset_allocator();
    }

    /// Spawn per-area Phase 2 (`FixUpRegion`) work packets into `WorkBucketStage::Compact`.
    pub fn add_fixup_tasks(&'static self) {
        fn inner<VM: VMBinding, const CAN_BMI2: bool>(this: &'static ParallelIBMMCSpace<VM>) {
            let packets: Vec<Box<dyn GCWork<VM>>> = this
                .forwarding
                .non_empty_ranges()
                .into_iter()
                .map(|(start, end)| {
                    Box::new(FixUpRegion::<VM, CAN_BMI2>::new(this, start, end))
                        as Box<dyn GCWork<VM>>
                })
                .collect();
            this.scheduler.work_buckets[WorkBucketStage::Compact].bulk_add(packets);
        }
        if self.forwarding.supports_bmi2() {
            inner::<VM, true>(self);
        } else {
            inner::<VM, false>(self);
        }
    }

    /// Phase 2 (Fixup Phase): scan relocated live objects in `[start, end)` via `ALLOC_SPEC`
    /// and update every in-space reference using the mark bitmap, offset vector, and alloc bitmap.
    pub fn fixup_region<const CAN_BMI2: bool>(
        &self,
        worker: &mut GCWorker<VM>,
        start: Address,
        end: Address,
    ) {
        if end <= start {
            return;
        }
        self.forwarding.scan_alloc_objects(start, end, &mut |object| {
            if VM::VMScanning::support_slot_enqueuing(worker.tls, object) {
                VM::VMScanning::scan_object(worker.tls, object, &mut |s: VM::VMSlot| {
                    if let Some(o) = s.load() {
                        if self.in_space(o) {
                            let new_o = self.forward::<CAN_BMI2>(o, true);
                            trace!("fixup slot: {} -> {}", o, new_o);
                            s.store(new_o);
                        }
                    }
                });
            } else {
                VM::VMScanning::scan_object_and_trace_edges(worker.tls, object, &mut |o| {
                    if self.in_space(o) {
                        self.forward::<CAN_BMI2>(o, true)
                    } else {
                        o
                    }
                });
            }
        });
    }

    fn update_slots<const CAN_BMI2: bool>(&self, slots: &[VM::VMSlot]) {
        for s in slots {
            if let Some(o) = s.load() {
                if self.in_space(o) {
                    let new_o = self.forward::<CAN_BMI2>(o, true);
                    trace!("forwarding remset {} -> {}", o, new_o);
                    s.store(new_o);
                }
            }
        }
    }
}

/// Phase 1 work packet: compact live objects across areas in [`ParallelIBMMCSpace`].
pub struct ParallelMove<VM: VMBinding> {
    space: &'static ParallelIBMMCSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for ParallelMove<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.space.parallel_move();
    }
}

impl<VM: VMBinding> ParallelMove<VM> {
    pub fn new(space: &'static ParallelIBMMCSpace<VM>) -> Self {
        Self { space }
    }
}

/// Sentinel work packet at the end of `CalculateForwarding` to update region cursors and mark forwarding ready.
pub struct AfterMove<VM: VMBinding> {
    space: &'static ParallelIBMMCSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for AfterMove<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.space.after_move();
    }
}

impl<VM: VMBinding> AfterMove<VM> {
    pub fn new(space: &'static ParallelIBMMCSpace<VM>) -> Self {
        Self { space }
    }
}

/// Phase 2 work packet: fix up all references inside compacted live objects in one area.
pub struct FixUpRegion<VM: VMBinding, const CAN_BMI2: bool> {
    space: &'static ParallelIBMMCSpace<VM>,
    start: Address,
    end: Address,
}

impl<VM: VMBinding, const CAN_BMI2: bool> GCWork<VM> for FixUpRegion<VM, CAN_BMI2> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.space
            .fixup_region::<CAN_BMI2>(worker, self.start, self.end);
    }
}

impl<VM: VMBinding, const CAN_BMI2: bool> FixUpRegion<VM, CAN_BMI2> {
    pub fn new(space: &'static ParallelIBMMCSpace<VM>, start: Address, end: Address) -> Self {
        Self { space, start, end }
    }
}

/// Update references in a vector of remembered external/root slots.
pub struct UpdateSlots<VM: VMBinding, const CAN_BMI2: bool> {
    space: &'static ParallelIBMMCSpace<VM>,
    slots: Vec<VM::VMSlot>,
}

impl<VM: VMBinding, const CAN_BMI2: bool> GCWork<VM> for UpdateSlots<VM, CAN_BMI2> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.space.update_slots::<CAN_BMI2>(&self.slots);
    }
}

impl<VM: VMBinding, const CAN_BMI2: bool> UpdateSlots<VM, CAN_BMI2> {
    pub fn new(space: &'static ParallelIBMMCSpace<VM>, slots: Vec<VM::VMSlot>) -> Self {
        Self { space, slots }
    }
}
