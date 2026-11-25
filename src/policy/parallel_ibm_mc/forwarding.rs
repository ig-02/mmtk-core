#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
use crate::util::constants::{
    BITS_IN_ADDRESS, BYTES_IN_ADDRESS, BYTES_IN_WORD, LOG_BITS_IN_ADDRESS, LOG_BITS_IN_BYTE,
    LOG_BYTES_IN_WORD,
};
#[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
use crate::util::constants::BYTES_IN_WORD;
use crate::util::linear_scan::Region;
use crate::util::metadata::side_metadata::spec_defs::{
    COMPRESSOR_MARK, IBM_MC_ALLOC, IBM_MC_OFFSET_VECTOR,
};
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::{Address, ObjectReference};
use crate::vm::VMBinding;
use atomic::Ordering;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Mutex;

/// A block in Parallel IBM MC is the granularity (256 bytes, following Abuaiadh et al., OOPSLA '04)
/// at which the offset vector records the relocated address of the first live object.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(crate) struct Block(Address);

impl Region for Block {
    const LOG_BYTES: usize = 8; // 256 bytes
    fn from_aligned_address(address: Address) -> Self {
        debug_assert!(address.is_aligned_to(Self::BYTES));
        Block(address)
    }
    fn start(&self) -> Address {
        self.0
    }
}

/// An area in Parallel IBM MC is the granularity (256 KiB, matching `CompressorRegion`)
/// at which parallel worker threads claim compaction units and share free-space targets.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(crate) struct Area(Address);

impl Region for Area {
    const LOG_BYTES: usize = 18; // 256 KiB
    fn from_aligned_address(address: Address) -> Self {
        debug_assert!(address.is_aligned_to(Self::BYTES));
        Area(address)
    }
    fn start(&self) -> Address {
        self.0
    }
}

pub(crate) const MARK_SPEC: SideMetadataSpec = COMPRESSOR_MARK;
pub(crate) const ALLOC_SPEC: SideMetadataSpec = IBM_MC_ALLOC;
pub(crate) const OFFSET_VECTOR_SPEC: SideMetadataSpec = IBM_MC_OFFSET_VECTOR;

/// Shift converting a heap address to a byte offset in a 1-bit-per-word bitmap (MARK / ALLOC).
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
const LOG_DATA_TO_BIT_META_BYTE_SHIFT: usize =
    (LOG_BYTES_IN_WORD as usize) + (LOG_BITS_IN_BYTE as usize);

/// Shift converting a 256B-aligned block address to a byte offset in the offset vector.
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
const LOG_DATA_TO_OFFSET_META_BYTE_SHIFT: usize =
    Block::LOG_BYTES - (LOG_BYTES_IN_WORD as usize);

/// Number of heap bytes covered by one `usize` word of `ALLOC_SPEC` / `MARK_SPEC` (512 bytes on 64-bit).
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
const BYTES_PER_BITMAP_WORD: usize = BITS_IN_ADDRESS * BYTES_IN_WORD;
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
const LOG_BYTES_PER_BITMAP_WORD: usize = LOG_BITS_IN_ADDRESS + (LOG_BYTES_IN_WORD as usize);

/// Select the 0-indexed `k`-th set bit in `word` without BMI2.
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
fn select_nth_set_bit_fallback(mut word: usize, k: u32) -> u32 {
    debug_assert!(k < word.count_ones());
    for _ in 0..k {
        word &= word - 1;
    }
    word.trailing_zeros()
}

/// Select the 0-indexed `k`-th set bit in `word` using BMI2 `PDEP` + `TZCNT`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[target_feature(enable = "bmi2,popcnt")]
#[inline]
unsafe fn select_nth_set_bit_bmi2(word: usize, k: u32) -> u32 {
    debug_assert!(k < word.count_ones());
    let deposited = core::arch::x86_64::_pdep_u64(1u64 << k, word as u64);
    deposited.trailing_zeros()
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
fn select_nth_set_bit<const CAN_BMI2: bool>(word: usize, k: u32) -> u32 {
    if CAN_BMI2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            select_nth_set_bit_bmi2(word, k)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unreachable!("Shouldn't have CAN_BMI2 = true on non-x86_64")
        }
    } else {
        select_nth_set_bit_fallback(word, k)
    }
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
fn bit_meta_word_addr(base: Address, data_addr: Address) -> Address {
    let word_index = data_addr.as_usize() >> LOG_BYTES_PER_BITMAP_WORD;
    base + (word_index << (LOG_BYTES_IN_WORD as usize))
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
fn bit_meta_block_u32_addr(base: Address, block_start: Address) -> Address {
    debug_assert!(block_start.is_aligned_to(Block::BYTES));
    base + (block_start.as_usize() >> LOG_DATA_TO_BIT_META_BYTE_SHIFT)
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
fn offset_meta_addr(base: Address, block_start: Address) -> Address {
    debug_assert!(block_start.is_aligned_to(Block::BYTES));
    base + (block_start.as_usize() >> LOG_DATA_TO_OFFSET_META_BYTE_SHIFT)
}

/// Write-combined buffered writer for `ALLOC_SPEC` during the parallel Moving phase.
///
/// Unlike the single-threaded writer, a worker resuming compaction into an existing
/// target area at a mid-word `p_copy` must preserve alloc bits already written for
/// lower addresses in that first bitmap word. Subsequent words are known to be zero
/// because `prepare()` zeroes `ALLOC_SPEC` across every allocated region.
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
struct AllocBitmapWriter {
    alloc_base: Address,
    cur_word_meta_addr: Address,
    cur_word: usize,
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
impl AllocBitmapWriter {
    #[inline]
    fn new(alloc_base: Address) -> Self {
        Self {
            alloc_base,
            cur_word_meta_addr: Address::ZERO,
            cur_word: 0,
        }
    }

    #[inline(always)]
    unsafe fn set(&mut self, addr: Address) {
        let target_meta_addr = bit_meta_word_addr(self.alloc_base, addr);
        let bit = ((addr.as_usize() >> (LOG_BYTES_IN_WORD as usize)) & (BITS_IN_ADDRESS - 1)) as u32;
        if self.cur_word_meta_addr.is_zero() {
            // Preserve bits set by any previous worker that compacted into this target area.
            self.cur_word_meta_addr = target_meta_addr;
            self.cur_word = target_meta_addr.load::<usize>();
        } else if target_meta_addr != self.cur_word_meta_addr {
            self.cur_word_meta_addr.store::<usize>(self.cur_word);
            self.cur_word_meta_addr = target_meta_addr;
            self.cur_word = 0;
        }
        self.cur_word |= 1usize << bit;
    }

    #[inline]
    unsafe fn finish(self) {
        if !self.cur_word_meta_addr.is_zero() {
            self.cur_word_meta_addr.store::<usize>(self.cur_word);
        }
    }
}

/// Shared coordination state for Phase 1 (`Parallel_Move` in Abuaiadh et al., Fig. 2).
pub(crate) struct ParallelMoveState {
    regions: Vec<(Area, Address)>,
    last_compacted: AtomicUsize,
    /// `0` means NULL (locked or full/unclaimed); non-zero is the free-space `Address` in the area.
    target_table: Vec<AtomicUsize>,
    /// Final compacted top address for each area, used to update `RegionPageResource` and FixUp ranges.
    final_tops: Vec<AtomicUsize>,
}

impl ParallelMoveState {
    fn new(regions: Vec<(Area, Address)>) -> Self {
        let n = regions.len();
        let target_table = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let final_tops = regions
            .iter()
            .map(|(area, _)| AtomicUsize::new(area.start().as_usize()))
            .collect();
        Self {
            regions,
            last_compacted: AtomicUsize::new(0),
            target_table,
            final_tops,
        }
    }

    #[inline]
    fn acquire_target(&self, source_idx: usize) -> Option<(usize, Address)> {
        for t in 0..source_idx {
            // TTAS: avoid write invalidation on null/exhausted target entries.
            if self.target_table[t].load(Ordering::Relaxed) != 0 {
                let p_copy = self.target_table[t].swap(0, Ordering::AcqRel);
                if p_copy != 0 {
                    return Some((t, unsafe { Address::from_usize(p_copy) }));
                }
            }
        }
        None
    }

    #[inline]
    fn release_area_space(&self, idx: usize, top: Address, area_end: Address) {
        self.final_tops[idx].store(top.as_usize(), Ordering::Relaxed);
        if top < area_end {
            self.target_table[idx].store(top.as_usize(), Ordering::Release);
        }
    }
}

/// Forwarding metadata for the parallel IBM MC compactor.
pub struct ForwardingMetadata<VM: VMBinding> {
    calculated: AtomicBool,
    supports_bmi2: bool,
    move_state: Mutex<Option<ParallelMoveState>>,
    final_ranges: Mutex<Vec<(Address, Address)>>,
    vm: PhantomData<VM>,
}

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new() -> Self {
        cfg_if::cfg_if! { if #[cfg(target_arch = "x86_64")] {
            let supports_bmi2 = is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("popcnt");
        } else {
            let supports_bmi2 = false;
        }}
        ForwardingMetadata {
            calculated: AtomicBool::new(false),
            supports_bmi2,
            move_state: Mutex::new(None),
            final_ranges: Mutex::new(Vec::new()),
            vm: PhantomData,
        }
    }

    pub fn supports_bmi2(&self) -> bool {
        self.supports_bmi2
    }

    pub fn has_calculated_forwarding_addresses(&self) -> bool {
        self.calculated.load(Ordering::Relaxed)
    }

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
        *self.move_state.lock().unwrap() = None;
        self.final_ranges.lock().unwrap().clear();
    }

    /// Initialize shared `ParallelMoveState` before spawning `ParallelMove` workers.
    pub(crate) fn init_move_state(&self, regions: Vec<(Area, Address)>) {
        *self.move_state.lock().unwrap() = Some(ParallelMoveState::new(regions));
    }

    /// Execute the `Parallel_Move` loop for one worker using `measure_obj` and `move_obj`.
    ///
    /// - `measure_obj(obj, to)` computes the `next_to` cursor if `obj` were moved at `to`
    ///   (without copying or mutating object/VO state).
    /// - `move_obj(obj, to)` relocates `obj` at `to` and returns `(new_obj, next_to)`.
    pub(crate) fn run_parallel_move(
        &self,
        measure_obj: &mut impl FnMut(ObjectReference, Address) -> Address,
        move_obj: &mut impl FnMut(ObjectReference, Address) -> (ObjectReference, Address),
    ) {
        let guard = self.move_state.lock().unwrap();
        let state_ref: &ParallelMoveState = match guard.as_ref() {
            Some(s) => unsafe {
                // SAFETY: `ParallelMoveState` lives inside `self.move_state` and is only
                // replaced/cleared during `init_move_state` / `after_move` / `release`,
                // outside the parallel move bucket. Dropping the MutexGuard lets all workers
                // run `parallel_move_worker` concurrently on `state_ref`.
                &*(s as *const ParallelMoveState)
            },
            None => return,
        };
        drop(guard);

        #[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
        unsafe {
            parallel_move_worker_fast(
                state_ref,
                MARK_SPEC.get_absolute_offset(),
                ALLOC_SPEC.get_absolute_offset(),
                OFFSET_VECTOR_SPEC.get_absolute_offset(),
                measure_obj,
                move_obj,
            );
        }
        #[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
        {
            parallel_move_worker_fallback(state_ref, measure_obj, move_obj);
        }
    }

    /// Collect final compacted region tops after all `ParallelMove` workers complete,
    /// mark forwarding addresses ready, and return `(final_tops, non_empty_ranges)`.
    pub(crate) fn finish_move(&self) -> Vec<Address> {
        let state = self.move_state.lock().unwrap().take();
        let mut final_tops = Vec::new();
        let mut ranges = Vec::new();
        if let Some(state) = state {
            for ((area, _), top_atomic) in state.regions.iter().zip(state.final_tops.iter()) {
                let top = unsafe { Address::from_usize(top_atomic.load(Ordering::Relaxed)) };
                final_tops.push(top);
                if top > area.start() {
                    ranges.push((area.start(), top));
                }
            }
        }
        *self.final_ranges.lock().unwrap() = ranges;
        self.calculated.store(true, Ordering::Release);
        final_tops
    }

    /// Return the compacted `[start, top)` ranges of all non-empty areas for Phase 2 FixUp.
    pub(crate) fn non_empty_ranges(&self) -> Vec<(Address, Address)> {
        self.final_ranges.lock().unwrap().clone()
    }

    /// Scan relocated live objects in `[start, end)` using `ALLOC_SPEC`.
    pub fn scan_alloc_objects(
        &self,
        start: Address,
        end: Address,
        f: &mut impl FnMut(ObjectReference),
    ) {
        if end <= start {
            return;
        }
        #[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
        unsafe {
            scan_alloc_objects_fast(ALLOC_SPEC.get_absolute_offset(), start, end, f);
        }
        #[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
        {
            ALLOC_SPEC.scan_non_zero_values::<u8>(start, end, &mut |addr: Address| {
                f(ObjectReference::from_raw_address(addr).unwrap());
            });
        }
    }

    /// Compute the post-compaction address for an object whose pre-compaction address is `address`.
    #[inline(always)]
    pub fn forward<const CAN_BMI2: bool>(&self, address: Address) -> Address {
        debug_assert!(
            self.calculated.load(Ordering::Relaxed),
            "forward() should only be called after the Moving phase"
        );
        #[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
        unsafe {
            if CAN_BMI2 {
                #[cfg(target_arch = "x86_64")]
                {
                    forward_fast_bmi2(
                        MARK_SPEC.get_absolute_offset(),
                        ALLOC_SPEC.get_absolute_offset(),
                        OFFSET_VECTOR_SPEC.get_absolute_offset(),
                        address,
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    unreachable!("Shouldn't have CAN_BMI2 = true on non-x86_64")
                }
            } else {
                forward_fast::<false>(
                    MARK_SPEC.get_absolute_offset(),
                    ALLOC_SPEC.get_absolute_offset(),
                    OFFSET_VECTOR_SPEC.get_absolute_offset(),
                    address,
                )
            }
        }
        #[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
        {
            let _ = CAN_BMI2;
            self.forward_fallback(address)
        }
    }

    #[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
    fn forward_fallback(&self, address: Address) -> Address {
        let block = Block::from_unaligned_address(address);
        let mut ordinal: usize = 0;
        MARK_SPEC.scan_non_zero_values::<u8>(block.start(), address, &mut |_| {
            ordinal += 1;
        });
        let first_moved = unsafe {
            Address::from_usize(
                OFFSET_VECTOR_SPEC.load_atomic::<usize>(block.start(), Ordering::Relaxed),
            )
        };
        if ordinal == 0 {
            return first_moved;
        }
        let target_area = Area::from_unaligned_address(first_moved);
        let mut remaining = ordinal;
        let mut found = first_moved;
        let search_start = first_moved + BYTES_IN_WORD;
        let search_end = target_area.end();
        ALLOC_SPEC.scan_non_zero_values::<u8>(search_start, search_end, &mut |addr| {
            if remaining > 0 {
                remaining -= 1;
                if remaining == 0 {
                    found = addr;
                }
            }
        });
        debug_assert_eq!(
            remaining, 0,
            "Failed to locate {ordinal}-th object in alloc bitmap"
        );
        found
    }

    #[cfg(all(debug_assertions, feature = "vo_bit"))]
    pub fn scan_marked_objects(
        &self,
        start: Address,
        end: Address,
        f: &mut impl FnMut(ObjectReference),
    ) {
        MARK_SPEC.scan_non_zero_values::<u8>(start, end, &mut |addr: Address| {
            f(ObjectReference::from_raw_address(addr).unwrap());
        });
    }
}

/// Fast 64-bit check for whether `[start, end)` contains any set bit in `MARK_SPEC`.
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
unsafe fn area_has_live_objects_fast(mark_base: Address, start: Address, end: Address) -> bool {
    debug_assert!(start.is_aligned_to(BYTES_PER_BITMAP_WORD));
    let mut word_meta_addr = bit_meta_word_addr(mark_base, start);
    let mut word_data_base = start;
    let full_words_end = end.align_down(BYTES_PER_BITMAP_WORD);

    while word_data_base < full_words_end {
        if word_meta_addr.load::<usize>() != 0 {
            return true;
        }
        word_meta_addr += BYTES_IN_ADDRESS;
        word_data_base += BYTES_PER_BITMAP_WORD;
    }

    if word_data_base < end {
        let end_bit = (end - word_data_base) >> (LOG_BYTES_IN_WORD as usize);
        let mask = (1usize << end_bit) - 1;
        if (word_meta_addr.load::<usize>() & mask) != 0 {
            return true;
        }
    }
    false
}

/// Read the 32-bit mark mask for `block_start` (masked if `block_start + Block::BYTES > end`).
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
unsafe fn load_block_marks(mark_base: Address, block_start: Address, end: Address) -> u32 {
    let raw = bit_meta_block_u32_addr(mark_base, block_start).load::<u32>();
    let block_end = block_start + Block::BYTES;
    if block_end <= end {
        raw
    } else {
        let valid_bits = ((end - block_start) >> (LOG_BYTES_IN_WORD as usize)) as u32;
        if valid_bits == 0 {
            0
        } else {
            raw & ((1u32 << valid_bits) - 1)
        }
    }
}

/// Compute the resulting `to` cursor if all live objects in `block_start` (with mark mask `bits`)
/// are moved starting at `to`.
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
unsafe fn measure_block_end(
    block_start: Address,
    mut bits: u32,
    mut to: Address,
    measure_obj: &mut impl FnMut(ObjectReference, Address) -> Address,
) -> Address {
    while bits != 0 {
        let bit = bits.trailing_zeros();
        let obj_addr = block_start + ((bit as usize) << (LOG_BYTES_IN_WORD as usize));
        let obj = ObjectReference::from_raw_address_unchecked(obj_addr);
        to = measure_obj(obj, to);
        bits &= bits - 1;
    }
    to
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
unsafe fn move_block_marks(
    block_start: Address,
    mut bits: u32,
    mut to: Address,
    offset_base: Address,
    alloc_writer: &mut AllocBitmapWriter,
    move_obj: &mut impl FnMut(ObjectReference, Address) -> (ObjectReference, Address),
) -> Address {
    let first_bit = bits.trailing_zeros();
    let first_obj_addr = block_start + ((first_bit as usize) << (LOG_BYTES_IN_WORD as usize));
    let first_obj = ObjectReference::from_raw_address_unchecked(first_obj_addr);
    let (new_first_obj, next_to) = move_obj(first_obj, to);
    to = next_to;
    let new_first_addr = new_first_obj.to_raw_address();
    offset_meta_addr(offset_base, block_start).store::<usize>(new_first_addr.as_usize());
    alloc_writer.set(new_first_addr);
    bits &= bits - 1;

    while bits != 0 {
        let bit = bits.trailing_zeros();
        let obj_addr = block_start + ((bit as usize) << (LOG_BYTES_IN_WORD as usize));
        let obj = ObjectReference::from_raw_address_unchecked(obj_addr);
        let (new_obj, next_to) = move_obj(obj, to);
        to = next_to;
        alloc_writer.set(new_obj.to_raw_address());
        bits &= bits - 1;
    }
    to
}

/// Fast 64-bit `Parallel_Move` worker (Abuaiadh et al., Section 3 & Figure 2).
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
unsafe fn parallel_move_worker_fast(
    state: &ParallelMoveState,
    mark_base: Address,
    alloc_base: Address,
    offset_base: Address,
    measure_obj: &mut impl FnMut(ObjectReference, Address) -> Address,
    move_obj: &mut impl FnMut(ObjectReference, Address) -> (ObjectReference, Address),
) {
    let n = state.regions.len();
    loop {
        let s = state.last_compacted.fetch_add(1, Ordering::Relaxed);
        if s >= n {
            break;
        }
        let (s_area, s_cursor) = state.regions[s];
        let s_start = s_area.start();
        let s_end = s_area.end();

        if s_cursor <= s_start || !area_has_live_objects_fast(mark_base, s_start, s_cursor) {
            state.release_area_space(s, s_start, s_end);
            continue;
        }

        let mut block_cursor = s_start;
        while block_cursor < s_cursor {
            match state.acquire_target(s) {
                None => {
                    // Compact remaining blocks [block_cursor, s_cursor) internally into area `s`.
                    let mut to = s_start;
                    let mut alloc_writer = AllocBitmapWriter::new(alloc_base);
                    while block_cursor < s_cursor {
                        let bits = load_block_marks(mark_base, block_cursor, s_cursor);
                        if bits != 0 {
                            to = move_block_marks(
                                block_cursor,
                                bits,
                                to,
                                offset_base,
                                &mut alloc_writer,
                                move_obj,
                            );
                        }
                        block_cursor += Block::BYTES;
                    }
                    alloc_writer.finish();
                    state.release_area_space(s, to, s_end);
                    break;
                }
                Some((t, mut to)) => {
                    let t_end = state.regions[t].0.end();
                    let mut alloc_writer = AllocBitmapWriter::new(alloc_base);
                    let mut t_full = false;

                    while block_cursor < s_cursor {
                        let bits = load_block_marks(mark_base, block_cursor, s_cursor);
                        if bits != 0 {
                            let projected_to =
                                measure_block_end(block_cursor, bits, to, measure_obj);
                            if projected_to > t_end {
                                // Never split a 256B block across two areas: abandon `t`.
                                t_full = true;
                                break;
                            }
                            to = move_block_marks(
                                block_cursor,
                                bits,
                                to,
                                offset_base,
                                &mut alloc_writer,
                                move_obj,
                            );
                        }
                        block_cursor += Block::BYTES;
                    }

                    alloc_writer.finish();
                    if t_full {
                        // Target `t` stays NULL (`0`) in `target_table` so no other worker picks it.
                        state.final_tops[t].store(to.as_usize(), Ordering::Relaxed);
                    } else {
                        // Area `s` is now completely evacuated into lower areas.
                        state.release_area_space(t, to, t_end);
                        state.release_area_space(s, s_start, s_end);
                        break;
                    }
                }
            }
        }
    }
}

/// Fallback `Parallel_Move` worker for 32-bit platforms or `extreme_assertions`.
#[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
fn parallel_move_worker_fallback(
    state: &ParallelMoveState,
    measure_obj: &mut impl FnMut(ObjectReference, Address) -> Address,
    move_obj: &mut impl FnMut(ObjectReference, Address) -> (ObjectReference, Address),
) {
    let n = state.regions.len();
    loop {
        let s = state.last_compacted.fetch_add(1, Ordering::Relaxed);
        if s >= n {
            break;
        }
        let (s_area, s_cursor) = state.regions[s];
        let s_start = s_area.start();
        let s_end = s_area.end();

        let mut any_live = false;
        if s_cursor > s_start {
            MARK_SPEC.scan_non_zero_values::<u8>(s_start, s_cursor, &mut |_| {
                any_live = true;
            });
        }
        if !any_live {
            state.release_area_space(s, s_start, s_end);
            continue;
        }

        let mut block_cursor = s_start;
        while block_cursor < s_cursor {
            match state.acquire_target(s) {
                None => {
                    let mut to = s_start;
                    while block_cursor < s_cursor {
                        let block_end = (block_cursor + Block::BYTES).min(s_cursor);
                        let mut is_first = true;
                        MARK_SPEC.scan_non_zero_values::<u8>(
                            block_cursor,
                            block_end,
                            &mut |addr| {
                                let obj = ObjectReference::from_raw_address(addr).unwrap();
                                let (new_obj, next_to) = move_obj(obj, to);
                                to = next_to;
                                let new_addr = new_obj.to_raw_address();
                                if is_first {
                                    OFFSET_VECTOR_SPEC.store_atomic::<usize>(
                                        block_cursor,
                                        new_addr.as_usize(),
                                        Ordering::Relaxed,
                                    );
                                    is_first = false;
                                }
                                ALLOC_SPEC.store_atomic::<u8>(new_addr, 1, Ordering::Relaxed);
                            },
                        );
                        block_cursor += Block::BYTES;
                    }
                    state.release_area_space(s, to, s_end);
                    break;
                }
                Some((t, mut to)) => {
                    let t_end = state.regions[t].0.end();
                    let mut t_full = false;

                    while block_cursor < s_cursor {
                        let block_end = (block_cursor + Block::BYTES).min(s_cursor);
                        let mut projected_to = to;
                        let mut has_live = false;
                        MARK_SPEC.scan_non_zero_values::<u8>(
                            block_cursor,
                            block_end,
                            &mut |addr| {
                                has_live = true;
                                let obj = ObjectReference::from_raw_address(addr).unwrap();
                                projected_to = measure_obj(obj, projected_to);
                            },
                        );
                        if has_live {
                            if projected_to > t_end {
                                t_full = true;
                                break;
                            }
                            let mut is_first = true;
                            MARK_SPEC.scan_non_zero_values::<u8>(
                                block_cursor,
                                block_end,
                                &mut |addr| {
                                    let obj = ObjectReference::from_raw_address(addr).unwrap();
                                    let (new_obj, next_to) = move_obj(obj, to);
                                    to = next_to;
                                    let new_addr = new_obj.to_raw_address();
                                    if is_first {
                                        OFFSET_VECTOR_SPEC.store_atomic::<usize>(
                                            block_cursor,
                                            new_addr.as_usize(),
                                            Ordering::Relaxed,
                                        );
                                        is_first = false;
                                    }
                                    ALLOC_SPEC.store_atomic::<u8>(new_addr, 1, Ordering::Relaxed);
                                },
                            );
                        }
                        block_cursor += Block::BYTES;
                    }

                    if t_full {
                        state.final_tops[t].store(to.as_usize(), Ordering::Relaxed);
                    } else {
                        state.release_area_space(t, to, t_end);
                        state.release_area_space(s, s_start, s_end);
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
unsafe fn scan_alloc_objects_fast(
    alloc_base: Address,
    start: Address,
    end: Address,
    f: &mut impl FnMut(ObjectReference),
) {
    debug_assert!(start.is_aligned_to(BYTES_PER_BITMAP_WORD));
    let mut word_meta_addr = bit_meta_word_addr(alloc_base, start);
    let mut word_data_base = start;

    let full_words_end = end.align_down(BYTES_PER_BITMAP_WORD);
    while word_data_base < full_words_end {
        let mut word = word_meta_addr.load::<usize>();
        while word != 0 {
            let bit = word.trailing_zeros();
            let addr = word_data_base + ((bit as usize) << (LOG_BYTES_IN_WORD as usize));
            f(ObjectReference::from_raw_address_unchecked(addr));
            word &= word - 1;
        }
        word_meta_addr += BYTES_IN_ADDRESS;
        word_data_base += BYTES_PER_BITMAP_WORD;
    }

    if word_data_base < end {
        let end_bit = (end - word_data_base) >> (LOG_BYTES_IN_WORD as usize);
        debug_assert!(end_bit > 0 && end_bit < BITS_IN_ADDRESS);
        let mask = (1usize << end_bit) - 1;
        let mut word = word_meta_addr.load::<usize>() & mask;
        while word != 0 {
            let bit = word.trailing_zeros();
            let addr = word_data_base + ((bit as usize) << (LOG_BYTES_IN_WORD as usize));
            f(ObjectReference::from_raw_address_unchecked(addr));
            word &= word - 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[target_feature(enable = "bmi2,popcnt")]
unsafe fn forward_fast_bmi2(
    mark_base: Address,
    alloc_base: Address,
    offset_base: Address,
    address: Address,
) -> Address {
    forward_fast::<true>(mark_base, alloc_base, offset_base, address)
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
unsafe fn forward_fast<const CAN_BMI2: bool>(
    mark_base: Address,
    alloc_base: Address,
    offset_base: Address,
    address: Address,
) -> Address {
    let block_start = address.align_down(Block::BYTES);
    let bit_in_block = ((address - block_start) >> (LOG_BYTES_IN_WORD as usize)) as u32;
    let block_marks = bit_meta_block_u32_addr(mark_base, block_start).load::<u32>();
    // 0-indexed ordinal of this object within its 256B block (`i - 1` in Abuaiadh et al.).
    let rank = (block_marks & ((1u32 << bit_in_block) - 1)).count_ones();
    let o_addr = Address::from_usize(offset_meta_addr(offset_base, block_start).load::<usize>());

    // Fast path: the first live object in a block moves directly to `O`.
    if rank == 0 {
        return o_addr;
    }

    // Search `ALLOC_SPEC` starting strictly after `o_addr` for the `rank`-th (1-indexed) set bit.
    let mut word_meta_addr = bit_meta_word_addr(alloc_base, o_addr);
    let mut word_data_base = o_addr.align_down(BYTES_PER_BITMAP_WORD);
    let start_bit_in_word =
        ((o_addr - word_data_base) >> (LOG_BYTES_IN_WORD as usize)) as u32;
    // Clear bits `0..=start_bit_in_word` without shift overflow even when `start_bit_in_word == 63`.
    let mask = (!1usize) << start_bit_in_word;
    let mut word = word_meta_addr.load::<usize>() & mask;
    let mut remaining = rank;

    loop {
        let ones = word.count_ones();
        if remaining <= ones {
            let bit_in_word = select_nth_set_bit::<CAN_BMI2>(word, remaining - 1);
            return word_data_base + ((bit_in_word as usize) << (LOG_BYTES_IN_WORD as usize));
        }
        remaining -= ones;
        word_meta_addr += BYTES_IN_ADDRESS;
        word_data_base += BYTES_PER_BITMAP_WORD;
        word = word_meta_addr.load::<usize>();
    }
}

#[cfg(all(test, target_pointer_width = "64", not(feature = "extreme_assertions")))]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Barrier;

    #[test]
    fn test_alloc_bitmap_writer_preserves_existing_bits() {
        // Heap addresses [0, 2048) -> 4 alloc words.
        // Pre-populate word 0 with bits at offset 0 and 16 (simulating a previous thread).
        let mut alloc_buf = vec![0u64; 4];
        alloc_buf[0] = (1u64 << 0) | (1u64 << 2);
        let alloc_base = Address::from_mut_ptr(alloc_buf.as_mut_ptr());

        unsafe {
            let mut writer = AllocBitmapWriter::new(alloc_base);
            // Resume at offset 64 (bit 8 in word 0) and cross into word 1 (offset 528, bit 2).
            writer.set(Address::from_usize(64));
            writer.set(Address::from_usize(528));
            writer.finish();
        }

        assert_eq!(
            alloc_buf[0],
            (1u64 << 0) | (1u64 << 2) | (1u64 << 8),
            "Pre-existing lower bits in the starting bitmap word must be preserved"
        );
        assert_eq!(alloc_buf[1], 1u64 << 2);
        assert_eq!(alloc_buf[2], 0);
    }

    #[test]
    fn test_parallel_move_multithreaded_cross_area_and_non_splitting() {
        // Simulate 16 Areas (16 * 256 KiB = 4 MiB) at [0x4_0000, 0x44_0000).
        const NUM_AREAS: usize = 16;
        const HEAP_START: usize = Area::BYTES; // 256 KiB (aligned to Area::BYTES)
        const HEAP_BYTES: usize = NUM_AREAS * Area::BYTES;
        const HEAP_END: usize = HEAP_START + HEAP_BYTES;

        let mut mark_buf = vec![0u64; HEAP_END / BYTES_PER_BITMAP_WORD];
        let mut alloc_buf = vec![0u64; HEAP_END / BYTES_PER_BITMAP_WORD];
        let mut offset_buf = vec![0usize; HEAP_END / Block::BYTES];

        let mark_base = Address::from_mut_ptr(mark_buf.as_mut_ptr());
        let alloc_base = Address::from_mut_ptr(alloc_buf.as_mut_ptr());
        let offset_base = Address::from_mut_ptr(offset_buf.as_mut_ptr());

        let mut regions = Vec::new();
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        let mut all_live: Vec<(usize, usize)> = Vec::new();

        for a in 0..NUM_AREAS {
            let area_start = HEAP_START + a * Area::BYTES;
            let area = Area::from_aligned_address(unsafe { Address::from_usize(area_start) });
            let cursor = unsafe { Address::from_usize(area_start + Area::BYTES) };
            regions.push((area, cursor));

            // Leave areas 2 and 7 completely empty to exercise empty-area handoff.
            if a == 2 || a == 7 {
                continue;
            }

            // Place deterministic blocks of varied sizes across each non-empty area,
            // leaving ~50-75% free space so cross-area compaction & block non-splitting trigger.
            let mut blk = area_start;
            let mut idx = 0usize;
            while blk + Block::BYTES <= area_start + Area::BYTES {
                if (a + idx) % 3 != 0 {
                    // Two or three live objects per populated block, plus a trailing dense block.
                    let objs: &[(usize, usize)] = if idx % 5 == 1 {
                        &[(0, 128), (128, 128)]
                    } else if idx % 5 == 2 {
                        &[(8, 24), (48, 40), (192, 64)]
                    } else {
                        &[(16, 48), (120, 80)]
                    };
                    for &(off, sz) in objs {
                        let addr = blk + off;
                        let word_idx = addr >> LOG_BYTES_PER_BITMAP_WORD;
                        let bit = (addr >> (LOG_BYTES_IN_WORD as usize)) & (BITS_IN_ADDRESS - 1);
                        mark_buf[word_idx] |= 1u64 << bit;
                        sizes.insert(addr, sz);
                        all_live.push((addr, sz));
                    }
                }
                blk += Block::BYTES;
                idx += 1;
            }
        }

        let state = ParallelMoveState::new(regions.clone());
        let num_threads = 4;
        let barrier = Barrier::new(num_threads);
        let moved_records: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

        std::thread::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| {
                    barrier.wait();
                    let mut local_moves = Vec::new();
                    unsafe {
                        parallel_move_worker_fast(
                            &state,
                            mark_base,
                            alloc_base,
                            offset_base,
                            &mut |obj, to| {
                                let sz = sizes[&obj.to_raw_address().as_usize()];
                                to + sz
                            },
                            &mut |obj, to| {
                                let old_addr = obj.to_raw_address().as_usize();
                                let sz = sizes[&old_addr];
                                local_moves.push((old_addr, to.as_usize(), sz));
                                let new_obj = ObjectReference::from_raw_address_unchecked(to);
                                (new_obj, to + sz)
                            },
                        );
                    }
                    moved_records.lock().unwrap().extend(local_moves);
                });
            }
        });

        let moved = moved_records.into_inner().unwrap();
        assert_eq!(moved.len(), all_live.len(), "Every live object must be moved once");

        // 1. Verify no overlaps and every moved object stays inside a single area.
        let mut sorted_dest = moved.clone();
        sorted_dest.sort_by_key(|&(_, new_addr, _)| new_addr);
        for w in sorted_dest.windows(2) {
            let (_, a_start, a_sz) = w[0];
            let (_, b_start, _) = w[1];
            assert!(
                a_start + a_sz <= b_start,
                "Overlap detected: [{a_start:#x}, {:#x}) vs {b_start:#x}",
                a_start + a_sz
            );
        }
        for &(_, new_addr, sz) in &sorted_dest {
            let area_idx = (new_addr - HEAP_START) / Area::BYTES;
            let area_end = HEAP_START + (area_idx + 1) * Area::BYTES;
            assert!(
                new_addr + sz <= area_end,
                "Object at {new_addr:#x}+{sz} crossed area boundary {area_end:#x}"
            );
        }

        // 2. Verify invariant: compaction never moves objects to a higher-indexed area.
        for &(old_addr, new_addr, _) in &moved {
            let old_area = (old_addr - HEAP_START) / Area::BYTES;
            let new_area = (new_addr - HEAP_START) / Area::BYTES;
            assert!(
                new_area <= old_area,
                "Object moved to higher area: old_area={old_area}, new_area={new_area}"
            );
        }

        // 3. Verify block non-splitting: all live objects from the same 256B block land contiguously
        //    in the exact same destination area.
        let old_to_new: HashMap<usize, (usize, usize)> =
            moved.iter().map(|&(o, n, s)| (o, (n, s))).collect();
        let mut i = 0;
        while i < all_live.len() {
            let block_id = all_live[i].0 / Block::BYTES;
            let mut j = i + 1;
            while j < all_live.len() && all_live[j].0 / Block::BYTES == block_id {
                let (prev_new, prev_sz) = old_to_new[&all_live[j - 1].0];
                let (cur_new, _) = old_to_new[&all_live[j].0];
                assert_eq!(
                    prev_new + prev_sz,
                    cur_new,
                    "Block {block_id} was split or reordered!"
                );
                assert_eq!(
                    (prev_new - HEAP_START) / Area::BYTES,
                    (cur_new - HEAP_START) / Area::BYTES,
                    "Block {block_id} split across areas!"
                );
                j += 1;
            }
            i = j;
        }

        // 4. Verify `forward_fast` (fallback & BMI2) matches exact destination for every live object.
        #[cfg(target_arch = "x86_64")]
        let has_bmi2 = is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("popcnt");
        for &(old_addr, expected_new, _) in &moved {
            let got = unsafe {
                forward_fast::<false>(
                    mark_base,
                    alloc_base,
                    offset_base,
                    Address::from_usize(old_addr),
                )
            };
            assert_eq!(
                got.as_usize(),
                expected_new,
                "forward_fast::<false> mismatch for old={old_addr:#x}"
            );
            #[cfg(target_arch = "x86_64")]
            if has_bmi2 {
                let got_bmi2 = unsafe {
                    forward_fast_bmi2(
                        mark_base,
                        alloc_base,
                        offset_base,
                        Address::from_usize(old_addr),
                    )
                };
                assert_eq!(got_bmi2.as_usize(), expected_new);
            }
        }

        // 5. Verify `scan_alloc_objects_fast` across [area.start(), final_top) finds exact objects.
        let mut scanned = Vec::new();
        for (idx, (area, _)) in regions.iter().enumerate() {
            let top = unsafe { Address::from_usize(state.final_tops[idx].load(Ordering::Relaxed)) };
            assert!(top >= area.start() && top <= area.end());
            if top > area.start() {
                unsafe {
                    scan_alloc_objects_fast(alloc_base, area.start(), top, &mut |obj| {
                        scanned.push(obj.to_raw_address().as_usize());
                    });
                }
            }
        }
        let expected_sorted: Vec<usize> = sorted_dest.iter().map(|&(_, n, _)| n).collect();
        assert_eq!(scanned, expected_sorted);
    }
}
