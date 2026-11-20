use crate::sources::packed_digits::{PackedQuadPairs, PackedSignedDigits};
use jolt_field::Field;

pub(super) enum QuadClasses {
    Expanded(Vec<u16>),
    Packed {
        #[cfg(akita_fold_metal)]
        metal: Option<metal::MetalLookup>,
        witness: PackedSignedDigits,
    },
}

impl QuadClasses {
    pub(super) fn new<E: Field + 'static>(
        witness: &PackedSignedDigits,
        count: usize,
        metal_kernels: bool,
    ) -> Self {
        if metal_kernels
            && count >= 1 << 20
            && std::any::TypeId::of::<E>()
                == std::any::TypeId::of::<jolt_field::Ext2<jolt_field::Prime64Offset59>>()
            && PackedQuadPairs::new(witness.view()).is_some()
        {
            #[cfg(akita_fold_metal)]
            // SAFETY: The following variant retains the immutable witness and drops the handle first.
            let metal = unsafe { metal::MetalLookup::new(witness.encoded_bytes()) };
            return Self::Packed {
                #[cfg(akita_fold_metal)]
                metal,
                witness: witness.clone(),
            };
        }

        Self::Expanded(vec![0; count])
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Expanded(v) => v.len(),
            Self::Packed { witness, .. } => witness.len() / 4,
        }
    }
    pub(super) fn at(&self, i: usize) -> u16 {
        match self {
            Self::Expanded(v) => v[i],
            Self::Packed { witness, .. } => PackedQuadPairs::new(witness.view())
                .expect("validated packed quad witness")
                .at(i),
        }
    }
}

#[cfg(akita_fold_metal)]
#[path = "packed_lookup_metal.rs"]
mod metal;
