use crate::ddnnf::anomalies::t_wise_sampling::covering_strategies::cover_with_caching;
use crate::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
use crate::ddnnf::anomalies::t_wise_sampling::t_iterator::TInteractionIter;
use crate::ddnnf::anomalies::t_wise_sampling::sample_merger::{AndMerger, SampleMerger};
use crate::ddnnf::anomalies::t_wise_sampling::sat_wrapper::SatWrapper;
use crate::ddnnf::extended_ddnnf::objective_function::FloatOrd;
use std::cmp::min;
use itertools::Itertools;

use rand::prelude::StdRng;

use streaming_iterator::StreamingIterator;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;

#[derive(Debug, Clone)]
pub struct AttributeZippingMerger<'a> {
    pub t: usize,
    pub sat_solver: &'a SatWrapper<'a>,
    pub ext_ddnnf: &'a ExtendedDdnnf,
}

// Mark AttributeZippingMerger as an AndMerger
impl AndMerger for AttributeZippingMerger<'_> {}

impl SampleMerger for AttributeZippingMerger<'_> {
    fn merge(
        &self,
        node_id: usize,
        left: &Sample,
        right: &Sample,
        _rng: &mut StdRng,
    ) -> Sample {

        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut new_sample = AttributeZippingMerger::zip_samples(
            left,
            right,
            &self.ext_ddnnf
        );

        /*
        Iterate over the remaining interactions. Those are all interactions
        that contain at least one literal of the left and one of the right subgraph.
         */
        let left_literals: Vec<i32> = left.get_literals().to_vec();
        let right_literals: Vec<i32> = right.get_literals().to_vec();

        debug_assert!(!left_literals.iter().any(|x| *x == 0));
        debug_assert!(!right_literals.iter().any(|x| *x == 0));

        let mut interactions = Vec::new();


        // for k in 1..self.t {
        //     // take k literals of the left subgraph and t-k literals of the right subgraph
        //     let left_len = min(left_literals.len(), k);
        //     let right_len = min(right_literals.len(), self.t - k);
        //     //let left_iter = t_wise_over(left_literals, left_len);
        //     let mut left_iter = TInteractionIter::new(&left_literals, left_len);
        //
        //     while let Some(left_part) = left_iter.next() {
        //         //let right_iter = t_wise_over(right_literals, right_len);
        //         let mut right_iter =
        //             TInteractionIter::new(&right_literals, right_len);
        //         while let Some(right_part) = right_iter.next() {
        //             let mut interaction = right_part.to_vec();
        //             interaction.extend_from_slice(left_part);
        //             interactions.push(interaction);
        //         }
        //     }
        // }

        for k in 1..self.t {
            // take k literals of the left subgraph and t-k literals of the right subgraph
            let left_len = min(left_literals.len(), k);
            let right_len = min(right_literals.len(), self.t - k);

            TInteractionIter::new(&left_literals, left_len)
                .for_each(|left_part| {
                            TInteractionIter::new(&right_literals, right_len)
                                .for_each(|right_part| {
                                    let interaction = left_part.iter().chain(right_part.iter()).copied().collect_vec();
                                    interactions.push(interaction);
                                });
                        });
        }

        interactions.iter()
            .sorted_by_cached_key(|interaction| {
                FloatOrd::from(self.ext_ddnnf.get_objective_fn_val_of_literals(&interaction[..]))
            })
            .rev()
            .for_each(|interaction| {
                cover_with_caching(
                    &mut new_sample,
                    &interaction,
                    self.sat_solver,
                    node_id,
                    self.ext_ddnnf.ddnnf.number_of_variables as usize,
                );
                new_sample.partial_configs.sort_by_cached_key(|config|
                    FloatOrd::from(self.ext_ddnnf.get_average_objective_fn_val_of_config(config))
                );
            });

        new_sample
    }

    fn merge_all(
        &self,
        node_id: usize,
        samples: &[&Sample],
        rng: &mut StdRng,
    ) -> Sample {
        samples.iter()
            .sorted()
            .fold(
                Sample::default(),
                |acc, s| self.merge_in_place(node_id, acc, s, rng)
            )
    }
}


impl AttributeZippingMerger<'_> {
    fn zip_samples(
        left: &Sample,
        right: &Sample,
        ext_ddnnf: &ExtendedDdnnf,
    ) -> Sample {

        let mut new_sample = Sample::new_from_samples(&[left, right]);

        let left_sorted = left.iter_with_completeness()
            .sorted_by_cached_key(|(config, _)| {
                FloatOrd::from(ext_ddnnf.get_average_objective_fn_val_of_config(config))
            })
            .collect_vec();

        let right_sorted = right.iter_with_completeness()
            .sorted_by_cached_key(|(config, _)|
                FloatOrd::from(ext_ddnnf.get_average_objective_fn_val_of_config(config))
            )
            .collect_vec();

        left_sorted.iter().rev()
            .zip(right_sorted.iter().rev())
            .for_each(
                |(
                     (left_config, left_complete),
                     (right_config, right_complete),
                 )| {
                    let new_config = Config::from_disjoint(
                        left_config,
                        right_config,
                        ext_ddnnf.ddnnf.number_of_variables as usize,
                    );
                    if *left_complete && *right_complete {
                        new_sample.add_complete(new_config);
                    } else {
                        new_sample.add_partial(new_config);
                    }
                },
            );

        let remaining = if left.len() >= right.len() {
            left_sorted.into_iter().rev().skip(right.len())
        } else {
            right_sorted.into_iter().rev().skip(left.len())
        };
        remaining.for_each(|(config, _)| {
            new_sample.add_partial(config.clone())
            });

        new_sample
    }
}


#[cfg(test)]
mod test {

}
