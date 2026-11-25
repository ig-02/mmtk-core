#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
use crate::util::constants::{
    BITS_IN_ADDRESS, BYTES_IN_ADDRESS, BYTES_IN_WORD, LOG_BITS_IN_ADDRESS, LOG_BITS_IN_BYTE,
    LOG_BYTES_IN_WORD,
};
#[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
use crate::util::constants::BYTES_IN_WORD;
use crate::util::heap::MonotonePageResource;
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

/// A block in IBM MC is the granularity (256 bytes, following Abuaiadh et al., OOPSLA '04)
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

/// Write-combined buffered writer for `ALLOC_SPEC` during the single-threaded Moving phase.
///
/// Because compaction slides objects monotonically to increasing `to` addresses, this writer
/// accumulates bits for the current alloc bitmap word in registers, flushes on word transitions,
/// and zeroes any skipped or trailing words up to `new_top` without requiring a separate
/// `bzero_metadata` pass over the alloc bitmap.
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
struct AllocBitmapWriter {
    alloc_base: Address,
    cur_word_meta_addr: Address,
    cur_word: usize,
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
impl AllocBitmapWriter {
    #[inline]
    fn new(alloc_base: Address, first_address: Address) -> Self {
        debug_assert!(first_address.is_aligned_to(BYTES_PER_BITMAP_WORD));
        Self {
            alloc_base,
            cur_word_meta_addr: bit_meta_word_addr(alloc_base, first_address),
            cur_word: 0,
        }
    }

    #[inline(always)]
    unsafe fn set(&mut self, addr: Address) {
        let target_meta_addr = bit_meta_word_addr(self.alloc_base, addr);
        let bit = ((addr.as_usize() >> (LOG_BYTES_IN_WORD as usize)) & (BITS_IN_ADDRESS - 1)) as u32;
        if target_meta_addr != self.cur_word_meta_addr {
            self.cur_word_meta_addr.store::<usize>(self.cur_word);
            let gap_start = self.cur_word_meta_addr + BYTES_IN_ADDRESS;
            if target_meta_addr > gap_start {
                std::ptr::write_bytes(gap_start.to_mut_ptr::<u8>(), 0, target_meta_addr - gap_start);
            }
            self.cur_word_meta_addr = target_meta_addr;
            self.cur_word = 0;
        }
        self.cur_word |= 1usize << bit;
    }

    #[inline]
    unsafe fn finish(self, first_address: Address, new_top: Address) {
        if new_top <= first_address {
            return;
        }
        self.cur_word_meta_addr.store::<usize>(self.cur_word);
        let last_word_meta_addr = bit_meta_word_addr(self.alloc_base, new_top - 1usize);
        let gap_start = self.cur_word_meta_addr + BYTES_IN_ADDRESS;
        if last_word_meta_addr >= gap_start {
            let zero_bytes = (last_word_meta_addr - gap_start) + BYTES_IN_ADDRESS;
            std::ptr::write_bytes(gap_start.to_mut_ptr::<u8>(), 0, zero_bytes);
        }
    }
}

/// Forwarding metadata for the single-threaded IBM MC compactor.
pub struct ForwardingMetadata<VM: VMBinding> {
    pub(crate) first_address: Address,
    new_top: AtomicUsize,
    calculated: AtomicBool,
    supports_bmi2: bool,
    vm: PhantomData<VM>,
}

impl<VM: VMBinding> ForwardingMetadata<VM> {
    pub fn new(start: Address) -> Self {
        cfg_if::cfg_if! { if #[cfg(target_arch = "x86_64")] {
            let supports_bmi2 = is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("popcnt");
        } else {
            let supports_bmi2 = false;
        }}
        ForwardingMetadata {
            first_address: start,
            new_top: AtomicUsize::new(start.as_usize()),
            calculated: AtomicBool::new(false),
            supports_bmi2,
            vm: PhantomData,
        }
    }

    pub fn supports_bmi2(&self) -> bool {
        self.supports_bmi2
    }

    pub fn has_calculated_forwarding_addresses(&self) -> bool {
        self.calculated.load(Ordering::Relaxed)
    }

    pub fn new_top(&self) -> Address {
        unsafe { Address::from_usize(self.new_top.load(Ordering::Relaxed)) }
    }

    pub fn release(&self) {
        self.calculated.store(false, Ordering::Relaxed);
    }

    /// Run Phase 1 (Moving Phase) across `[first_address, pr.cursor())`.
    ///
    /// `move_obj` must relocate the given object to its destination and return
    /// `(new_obj, updated_to_cursor)`.
    pub fn move_objects(
        &self,
        pr: &MonotonePageResource<VM>,
        move_obj: &mut impl FnMut(ObjectReference) -> (ObjectReference, Address),
    ) -> Address {
        let start = self.first_address;
        let end = pr.cursor();
        let new_top = if end <= start {
            start
        } else {
            #[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
            unsafe {
                move_objects_fast_impl(
                    MARK_SPEC.get_absolute_offset(),
                    ALLOC_SPEC.get_absolute_offset(),
                    OFFSET_VECTOR_SPEC.get_absolute_offset(),
                    start,
                    end,
                    move_obj,
                )
            }
            #[cfg(any(not(target_pointer_width = "64"), feature = "extreme_assertions"))]
            {
                use crate::util::linear_scan::RegionIterator;
                let first_block = Block::from_aligned_address(start);
                let last_block = Block::from_aligned_address(end);
                let mut to = start;
                for block in RegionIterator::<Block>::new(first_block, last_block) {
                    let mut is_first = true;
                    MARK_SPEC.scan_non_zero_values::<u8>(
                        block.start(),
                        block.end(),
                        &mut |addr: Address| {
                            let obj = ObjectReference::from_raw_address(addr).unwrap();
                            let (new_obj, next_to) = move_obj(obj);
                            to = next_to;
                            let new_addr = new_obj.to_raw_address();
                            if is_first {
                                OFFSET_VECTOR_SPEC.store_atomic::<usize>(
                                    block.start(),
                                    new_addr.as_usize(),
                                    Ordering::Relaxed,
                                );
                                is_first = false;
                            }
                            ALLOC_SPEC.store_atomic::<u8>(new_addr, 1, Ordering::Relaxed);
                        },
                    );
                }
                to
            }
        };

        self.new_top.store(new_top.as_usize(), Ordering::Relaxed);
        self.calculated.store(true, Ordering::Relaxed);
        new_top
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
        let mut remaining = ordinal;
        let mut found = first_moved;
        let search_start = first_moved + BYTES_IN_WORD;
        let search_end = self.new_top();
        ALLOC_SPEC.scan_non_zero_values::<u8>(search_start, search_end, &mut |addr| {
            if remaining > 0 {
                remaining -= 1;
                if remaining == 0 {
                    found = addr;
                }
            }
        });
        debug_assert_eq!(remaining, 0, "Failed to locate {ordinal}-th object in alloc bitmap");
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

/// Fast 64-bit Moving-phase scan over 512B bitmap words (two 256B blocks per `u64`).
#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
unsafe fn move_objects_fast_impl(
    mark_base: Address,
    alloc_base: Address,
    offset_base: Address,
    start: Address,
    end: Address,
    move_obj: &mut impl FnMut(ObjectReference) -> (ObjectReference, Address),
) -> Address {
    debug_assert!(start.is_aligned_to(BYTES_PER_BITMAP_WORD));
    let mut alloc_writer = AllocBitmapWriter::new(alloc_base, start);
    let mut to = start;

    let mut block_pair_addr = start;
    while block_pair_addr < end {
        let mark_word_addr = bit_meta_word_addr(mark_base, block_pair_addr);
        let mark_word = mark_word_addr.load::<u64>();
        if mark_word != 0 {
            // Process first 256B block in the pair (low 32 bits).
            let low = mark_word as u32;
            if low != 0 {
                to = process_block_marks(
                    block_pair_addr,
                    low,
                    offset_base,
                    &mut alloc_writer,
                    move_obj,
                );
            }
            // Process second 256B block in the pair (high 32 bits).
            let second_block = block_pair_addr + Block::BYTES;
            if second_block < end {
                let high = (mark_word >> 32) as u32;
                if high != 0 {
                    to = process_block_marks(
                        second_block,
                        high,
                        offset_base,
                        &mut alloc_writer,
                        move_obj,
                    );
                }
            }
        }
        block_pair_addr += BYTES_PER_BITMAP_WORD;
    }

    alloc_writer.finish(start, to);
    to
}

#[cfg(all(target_pointer_width = "64", not(feature = "extreme_assertions")))]
#[inline(always)]
unsafe fn process_block_marks(
    block_start: Address,
    mut bits: u32,
    offset_base: Address,
    alloc_writer: &mut AllocBitmapWriter,
    move_obj: &mut impl FnMut(ObjectReference) -> (ObjectReference, Address),
) -> Address {
    // First live object in the block: record its relocated address in the offset vector
    // without an extra branch inside the loop.
    let first_bit = bits.trailing_zeros();
    let first_obj_addr = block_start + ((first_bit as usize) << (LOG_BYTES_IN_WORD as usize));
    let first_obj = ObjectReference::from_raw_address_unchecked(first_obj_addr);
    let (new_first_obj, mut to) = move_obj(first_obj);
    let new_first_addr = new_first_obj.to_raw_address();
    offset_meta_addr(offset_base, block_start).store::<usize>(new_first_addr.as_usize());
    alloc_writer.set(new_first_addr);
    bits &= bits - 1;

    while bits != 0 {
        let bit = bits.trailing_zeros();
        let obj_addr = block_start + ((bit as usize) << (LOG_BYTES_IN_WORD as usize));
        let obj = ObjectReference::from_raw_address_unchecked(obj_addr);
        let (new_obj, next_to) = move_obj(obj);
        to = next_to;
        alloc_writer.set(new_obj.to_raw_address());
        bits &= bits - 1;
    }
    to
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

    #[test]
    fn test_select_nth_set_bit() {
        let patterns: &[usize] = &[
            1,
            1usize << 63,
            0x8000_0000_0000_0001,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0x0F0F_00FF_1234_5678,
            usize::MAX,
        ];
        #[cfg(target_arch = "x86_64")]
        let has_bmi2 = is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("popcnt");

        for &w in patterns {
            let mut expected_positions = Vec::new();
            for bit in 0..BITS_IN_ADDRESS {
                if (w >> bit) & 1 != 0 {
                    expected_positions.push(bit as u32);
                }
            }
            for (k, &expected) in expected_positions.iter().enumerate() {
                assert_eq!(select_nth_set_bit_fallback(w, k as u32), expected);
                #[cfg(target_arch = "x86_64")]
                if has_bmi2 {
                    assert_eq!(unsafe { select_nth_set_bit_bmi2(w, k as u32) }, expected);
                }
            }
        }
    }

    #[test]
    fn test_alloc_bitmap_writer_zeroes_gaps_and_tail() {
        // Heap addresses [0, 8192) -> 16 alloc words = 128 bytes.
        let mut alloc_buf = vec![u64::MAX; 16];
        let alloc_base = Address::from_mut_ptr(alloc_buf.as_mut_ptr());
        let first = Address::ZERO;

        unsafe {
            let mut writer = AllocBitmapWriter::new(alloc_base, first);
            // Word 0: bits at 0 and 64 bytes (bit 0 and bit 8)
            writer.set(Address::from_usize(0));
            writer.set(Address::from_usize(64));
            // Skip words 1 and 2, set at 1536 bytes (word 3, bit 0)
            writer.set(Address::from_usize(1536));
            // End at 2600 bytes (inside word 5), leaving word 4 and 5 trailing zeros.
            writer.finish(first, Address::from_usize(2600));
        }

        assert_eq!(alloc_buf[0], (1u64 << 0) | (1u64 << 8));
        assert_eq!(alloc_buf[1], 0);
        assert_eq!(alloc_buf[2], 0);
        assert_eq!(alloc_buf[3], 1);
        assert_eq!(alloc_buf[4], 0);
        assert_eq!(alloc_buf[5], 0);
        // Words past new_top (word 6+) were not touched (still 0xFF..FF, outside scan range).
        assert_eq!(alloc_buf[6], u64::MAX);
    }

    #[test]
    fn test_move_scan_and_forward_end_to_end() {
        // Simulate a 16 KiB heap at virtual range [0x1000, 0x5000).
        const HEAP_START: usize = 0x1000;
        const HEAP_BYTES: usize = 16 * 1024;
        const HEAP_END: usize = HEAP_START + HEAP_BYTES;
        let mut mark_buf = vec![0u64; HEAP_END / BYTES_PER_BITMAP_WORD];
        let mut alloc_buf = vec![u64::MAX; HEAP_END / BYTES_PER_BITMAP_WORD]; // pre-dirty!
        let mut offset_buf = vec![usize::MAX; HEAP_END / Block::BYTES]; // pre-dirty!

        let mark_base = Address::from_mut_ptr(mark_buf.as_mut_ptr());
        let alloc_base = Address::from_mut_ptr(alloc_buf.as_mut_ptr());
        let offset_base = Address::from_mut_ptr(offset_buf.as_mut_ptr());

        // Construct test objects: (relative_old_addr, size)
        // Includes:
        // - unmoved prefix at HEAP_START
        // - dead holes
        // - dense 256B block with 16 objects of 16B each (ranks 0..15)
        // - empty blocks
        // - large 2048B object spanning multiple blocks
        // - object whose relocated `O` lands at bit 63 of an alloc word (512 - 8 = 504)!
        let live_objects: Vec<(usize, usize)> = vec![
            (HEAP_START + 0, 24),   // Block 0, rank 0 -> unmoved
            (HEAP_START + 48, 24),  // Block 0, rank 1 -> +24
            (HEAP_START + 200, 16), // Block 0, rank 2 -> +48
            // Block 1 (256..512): 16 live 16B objects packed
            (HEAP_START + 256, 16),
            (HEAP_START + 272, 16),
            (HEAP_START + 288, 16),
            (HEAP_START + 304, 16),
            (HEAP_START + 320, 16),
            (HEAP_START + 336, 16),
            (HEAP_START + 352, 16),
            (HEAP_START + 368, 16),
            (HEAP_START + 384, 16),
            (HEAP_START + 400, 16),
            (HEAP_START + 416, 16),
            (HEAP_START + 432, 16),
            (HEAP_START + 448, 16),
            (HEAP_START + 464, 16),
            (HEAP_START + 480, 16),
            (HEAP_START + 496, 16),
            // Current offset = 64 + 256 = 320.
            // Block 2 (512..768): one 184B object -> offset becomes 320 + 184 = 504!
            (HEAP_START + 520, 184),
            // Block 3 (768..1024) is empty!
            // Block 4 (1024..1280): first object moves to +504 (bit 63 of alloc word!),
            // second object moves to +520 (bit 1 of next alloc word!).
            (HEAP_START + 1024, 16),
            (HEAP_START + 1104, 32),
            // Multi-block 2048B object in Block 5 (1280..3328), followed by another object in Block 14 (3584)
            (HEAP_START + 1280, 2048),
            (HEAP_START + 3584, 64),
        ];

        // Populate mark bitmap (first word only).
        for &(addr, _) in &live_objects {
            let word_idx = addr >> LOG_BYTES_PER_BITMAP_WORD;
            let bit = (addr >> (LOG_BYTES_IN_WORD as usize)) & (BITS_IN_ADDRESS - 1);
            mark_buf[word_idx] |= 1u64 << bit;
        }

        // Compute expected forwarding map and run `move_objects_fast_impl`.
        let mut expected: Vec<(usize, usize)> = Vec::new();
        let mut sim_to = HEAP_START;
        for &(addr, size) in &live_objects {
            expected.push((addr, sim_to));
            sim_to += size;
        }

        let start = unsafe { Address::from_usize(HEAP_START) };
        let end = unsafe { Address::from_usize(HEAP_END) };
        let mut to = start;
        let mut moved_trace = Vec::new();
        let new_top = unsafe {
            move_objects_fast_impl(
                mark_base,
                alloc_base,
                offset_base,
                start,
                end,
                &mut |obj| {
                    let old_addr = obj.to_raw_address().as_usize();
                    let size = live_objects
                        .iter()
                        .find(|&&(a, _)| a == old_addr)
                        .expect("unexpected marked address")
                        .1;
                    let new_obj = ObjectReference::from_raw_address_unchecked(to);
                    moved_trace.push((old_addr, to.as_usize()));
                    to += size;
                    (new_obj, to)
                },
            )
        };

        assert_eq!(new_top.as_usize(), sim_to);
        assert_eq!(moved_trace, expected);

        // Verify `forward_fast` for every live object (both fallback and BMI2 if supported).
        #[cfg(target_arch = "x86_64")]
        let has_bmi2 = is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("popcnt");
        for &(old_addr, new_addr) in &expected {
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
                new_addr,
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
                assert_eq!(
                    got_bmi2.as_usize(),
                    new_addr,
                    "forward_fast_bmi2 mismatch for old={old_addr:#x}"
                );
            }
        }

        // Verify `scan_alloc_objects_fast` visits exact relocated addresses in order.
        let mut scanned = Vec::new();
        unsafe {
            scan_alloc_objects_fast(alloc_base, start, new_top, &mut |obj| {
                scanned.push(obj.to_raw_address().as_usize());
            });
        }
        let expected_new: Vec<usize> = expected.iter().map(|&(_, n)| n).collect();
        assert_eq!(scanned, expected_new);
    }

    #[test]
    fn test_empty_heap_move_and_scan() {
        const HEAP_BYTES: usize = 4096;
        let mark_buf = vec![0u64; HEAP_BYTES / BYTES_PER_BITMAP_WORD];
        let mut alloc_buf = vec![u64::MAX; HEAP_BYTES / BYTES_PER_BITMAP_WORD];
        let mut offset_buf = vec![usize::MAX; HEAP_BYTES / Block::BYTES];

        let mark_base = Address::from_ptr(mark_buf.as_ptr());
        let alloc_base = Address::from_mut_ptr(alloc_buf.as_mut_ptr());
        let offset_base = Address::from_mut_ptr(offset_buf.as_mut_ptr());

        let start = Address::ZERO;
        let end = unsafe { Address::from_usize(HEAP_BYTES) };
        let mut count = 0;
        let new_top = unsafe {
            move_objects_fast_impl(
                mark_base,
                alloc_base,
                offset_base,
                start,
                end,
                &mut |_| {
                    count += 1;
                    unreachable!("no live objects")
                },
            )
        };
        assert_eq!(count, 0);
        assert_eq!(new_top, start);

        let mut scanned = 0;
        unsafe {
            scan_alloc_objects_fast(alloc_base, start, new_top, &mut |_| {
                scanned += 1;
            });
        }
        assert_eq!(scanned, 0);
    }
}
