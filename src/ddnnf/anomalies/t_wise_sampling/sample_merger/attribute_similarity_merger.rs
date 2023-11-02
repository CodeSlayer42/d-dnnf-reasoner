use crate::ddnnf::anomalies::t_wise_sampling::data_structure::Sample;
use crate::ddnnf::anomalies::t_wise_sampling::sample_merger::{OrMerger, SampleMerger};
use rand::prelude::StdRng;
use itertools::Itertools;

use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use crate::ddnnf::extended_ddnnf::objective_function::FloatOrd;

#[derive(Debug, Copy, Clone)]
pub struct AttributeSimilarityMerger<'a> {
    pub t: usize,
    pub ext_ddnnf: &'a ExtendedDdnnf,
}

// Mark AttributeSimilarityMerger as an OrMerger
impl OrMerger for AttributeSimilarityMerger<'_> {}

impl SampleMerger for AttributeSimilarityMerger<'_> {
    fn merge<'a>(
        &self,
        _node_id: usize,
        left: &Sample,
        right: &Sample,
        _rng: &mut StdRng,
    ) -> Sample {

        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut new_sample = Sample::new_from_samples(&[left, right]);
        let candidates = left
            .iter()
            .chain(right.iter())
            .sorted_by_cached_key(|config|
                FloatOrd::from(self.ext_ddnnf.get_average_objective_fn_val_of_config(config))
            )
            .rev();

        for candidate in candidates {
            if new_sample.is_t_wise_covered(candidate, self.t) {
                continue;
            }

            if candidate.is_complete() {
                new_sample.add_complete(candidate.clone());
            } else {
                new_sample.add_partial(candidate.clone());
            }
        }

        new_sample
    }
}


#[cfg(test)]
mod test {

}
