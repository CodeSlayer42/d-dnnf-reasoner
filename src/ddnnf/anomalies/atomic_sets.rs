use bitvec::prelude::*;
use itertools::Itertools;
use rug::Integer;
use rustc_hash::FxHashMap;

use crate::Ddnnf;

use std::hash::Hash;

/// A quite basic union-find implementation that uses ranks and path compresion
#[derive(Debug, Clone, PartialEq)]
struct UnionFind<N: Hash + Eq + Clone> {
    size: usize,
    parents: FxHashMap<N, N>,
    rank: FxHashMap<N, usize>,
}

trait UnionFindTrait<N: Eq + Hash + Clone> {
    fn find(&mut self, node: N) -> N;
    fn equiv(&mut self, x: N, y: N) -> bool;
    fn union(&mut self, x: N, y: N);
    fn subsets(&mut self) -> Vec<Vec<N>>;
}

impl<T> Default for UnionFind<T>
where T: Eq + Hash + Clone {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> UnionFind<T>
where T: Eq + Hash + Clone {
    fn new() -> UnionFind<T> {
        let parents: FxHashMap<T, T> = FxHashMap::default();
        let rank: FxHashMap<T, usize> = FxHashMap::default();

        UnionFind {
            size: 0,
            parents,
            rank,
        }
    }
}

impl<T> UnionFindTrait<T> for UnionFind<T>
where T: Eq + Hash + Ord + Clone {
    // find(x): For x ∈ S, determines the unique representative to whose class x belongs.
    fn find(&mut self, node: T) -> T {
        if !self.parents.contains_key(&node) {
            self.parents.insert(node.clone(), node.clone());
            self.size += 1;
        }

        if !(node.eq(self.parents.get(&node).unwrap())) {
            let found = self.find((*self.parents.get(&node).unwrap()).clone());
            self.parents.insert(node.clone(), found);
        }
        (*self.parents.get(&node).unwrap()).clone()
    }

    // checks wether two values x and y share the same root
    fn equiv(&mut self, x: T, y: T) -> bool {
        self.find(x) == self.find(y)
    }

    // union(r, s): Unions the two classes belonging to the two representatives r and s,
    // and makes r the new representative of the new class.
    fn union(&mut self, x: T, y: T) {
        // Add "empty" rank information
        let x_root = self.find(x);
        let y_root = self.find(y);

        if !self.rank.contains_key(&x_root) {
            self.rank.insert(x_root.clone(), 0);
        }

        if !self.rank.contains_key(&y_root) {
            self.rank.insert(y_root.clone(), 0);
        }
        if x_root.eq(&y_root) {
            return;
        }

        // union by rank: The tree with the lower rank is always appended to the one with the higher rank
        let x_root_rank: usize = *self.rank.get(&x_root).unwrap();
        let y_root_rank: usize = *self.rank.get(&y_root).unwrap();

        if x_root_rank > y_root_rank {
            self.parents.insert(y_root, x_root);
        } else {
            self.parents.insert(x_root, y_root.clone());
            if x_root_rank == y_root_rank {
                self.rank.insert(y_root, y_root_rank + 1);
            }
        }
    }

    // computes all subsets by grouping values with the same root
    fn subsets(&mut self) -> Vec<Vec<T>> {
        let mut result: FxHashMap<T, Vec<T>> = FxHashMap::default();

        let rank_cp = self.rank.clone();
    
        for (node, _) in rank_cp.iter() {
            let root = self.find((*node).clone());

            if let std::collections::hash_map::Entry::Vacant(e) = result.entry(root.clone()) {
                let vec = vec![(*node).clone()];
                e.insert(vec);
            } else {
                let prev = &mut *result.get_mut(&root).unwrap();
                prev.push((*node).clone());
            }
        }
        let mut sets: Vec<Vec<_>> = result.into_values().collect();
        sets.iter_mut().for_each(|set| set.sort_unstable());
        sets
    }
}

impl Ddnnf {
    /// Compute all atomic sets
    /// A group forms an atomic set iff every valid configuration either includes
    /// or excludes all mebers of that atomic set
    pub(crate) fn get_atomic_sets(&mut self, candidates: Option<Vec<u32>>, assumptions: &[i32]) -> Vec<Vec<u16>> {
        let mut combinations: Vec<(Integer, i32)> = Vec::new();

        // If there are no candidates supplied, we consider all features to be a candidate
        let considered_features = match candidates {
            Some(c) => c,
            None => (1..=self.number_of_variables).collect(),
        };

        // We can't find any atomic set if there are no candidates
        if considered_features.is_empty() {
            return vec![];
        }

        // compute the cardinality of features to obtain atomic set candidates
        for i in considered_features {
            combinations.push((self.execute_query(&[&[i as i32], assumptions].concat()), i as i32));
        }
        combinations.sort_unstable(); // sorting is required to group in the next step
    
        // Group the features by their cardinality of feature count.
        // Features with the same count will be placed in the same group.
        let mut data_grouped = Vec::new();
        let mut current_key = combinations[0].0.clone();
        let mut values_current_key = Vec::new();
        for (key, value) in combinations.into_iter() {
            if current_key == key {
                values_current_key.push(value);
            } else {
                data_grouped.push((current_key, std::mem::take(&mut values_current_key)));
                current_key = key;
                values_current_key.push(value);
            }
        }
        data_grouped.push((current_key, values_current_key));

        // initalize Unionfind and Samples
        let mut atomic_sets: UnionFind<u16> = UnionFind::default();
        let signed_excludes = self.get_signed_excludes(assumptions);
        for (key, group) in data_grouped {
            self.incremental_subset_check(
                key, &group, &signed_excludes, assumptions, &mut atomic_sets);
        }

        let mut subsets = atomic_sets.subsets();
        subsets.sort_unstable();
        subsets
    }

    /// Computes the signs of the features in multiple uniform random samples.
    /// Each of the features is represented by an BitArray holds as many entries as random samples
    /// with a 0 indicating that the feature occurs negated and a 1 indicating the feature occurs affirmed.
    fn get_signed_excludes(&mut self, assumptions: &[i32]) -> Vec<BitArray<[u64; 8]>> {
        const SAMPLE_AMOUNT: usize = 512;

        let mut signed_excludes = Vec::with_capacity(self.number_of_variables as usize);

        let samples;
        match self.uniform_random_sampling(assumptions, SAMPLE_AMOUNT, 10) {
            Some(x) => samples = x,
            None => {
                // If the assumptions make the query unsat, then we get no samples.
                // Hence, we can't exclude any combination of features
                for _ in 0..self.number_of_variables as usize {
                    signed_excludes.push(bitarr![u64, Lsb0; 0; SAMPLE_AMOUNT]);
                }
                return signed_excludes;
            },
        }

        for var in 0..self.number_of_variables as usize {
            let mut bitvec = bitarr![u64, Lsb0; 0; SAMPLE_AMOUNT];
            for sample in samples.iter().enumerate() {
                bitvec.set(sample.0, sample.1[var].is_positive());
            }
            signed_excludes.push(bitvec);
        }

        signed_excludes
    }

    /// First naive approach to compute atomic sets by incrementally add a feature one by one
    /// while checking if the atomic set property (i.e. the count stays the same) still holds
    fn incremental_subset_check(&mut self, control: Integer, pot_atomic_set: &[i32],
        signed_excludes: &[BitArray<[u64; 8]>], assumptions: &[i32], atomic_sets: &mut UnionFind<u16>) {
        // goes through all combinations of set candidates and checks whether the pair is part of an atomic set
        for pair in pot_atomic_set.iter().copied().combinations(2) {
            let x = pair[0] as u16; let y = pair[1] as u16;

            // we don't have to check if a pair is part of an atomic set if they already are connected via transitivity
            if atomic_sets.equiv(x, y) {
                continue;
            }

            // If the sign of the two feature candidates differs in at least one of the uniform random samples,
            // then we can by sure that they don't belong to the same atomic set. Differences can be checked by
            // applying XOR to the two bitvectors and checking if any bit is set.
            if (signed_excludes[x as usize - 1] ^ signed_excludes[y as usize - 1]).any() {
                continue;
            }

            // we identify a pair of values to be in the same atomic set, then we union them
            if self.execute_query(&[&pair, assumptions].concat()) == control {
                atomic_sets.union(x, y);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashSet, iter::FromIterator};

    use crate::parser::build_ddnnf;

    use super::*;

    #[test]
    fn atomic_sets_vp9() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        // make sure that the results are reproducible
        for _ in 0..3 {
            let vp9_atomic_sets = vp9.get_atomic_sets(None, &vec![]);
            assert_eq!(vec![vec![1, 2, 6, 10, 15, 19, 25, 31, 40]], vp9_atomic_sets);

            // There should exactly one atomic set that is a subset of the core and the dead features.
            // Vp9 has no dead features. Hence, we can not test for a subset
            let vp9_core_features = HashSet::<_>::from_iter(vp9.core.iter().copied()).into_iter().map(|f| f as u16).collect::<Vec<u16>>();
            assert!(
                vp9_atomic_sets.iter()
                .filter(|set| vp9_core_features.iter().all(|f| set.contains(&f)))
                .exactly_one().is_ok()
            );
        
        }
    }

    #[test]
    fn atomic_sets_auto1() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        // ensure reproducible
        for _ in 0..3 {
            let auto1_atomic_sets = auto1.get_atomic_sets(None, &vec![]);
            assert_eq!(155, auto1_atomic_sets.len());

            // check some subset values
            assert_eq!(vec![1, 697, 1262], auto1_atomic_sets[0]);
            assert_eq!(vec![6, 76], auto1_atomic_sets[1]);
            assert_eq!(
                vec![20, 67, 68, 87, 106, 141, 154, 163, 165, 169, 394, 499, 564, 569, 570, 576, 591,
                    613, 626, 627, 629, 647, 648, 653, 696, 714, 724, 758, 868, 876, 935, 939, 940, 941,
                    1039, 1044, 1045, 1055, 1078, 1085, 1101, 1103, 1105, 1115, 1117, 1119, 1127, 1133,
                    1140, 1150, 1152, 1179, 1186, 1194, 1204, 1213, 1223, 1250, 1261, 1301, 1323, 1324,
                    1325, 1498, 1501, 1521, 1549, 1553, 1667, 1675, 1678, 1715, 1748, 1749, 1788, 1797,
                    1799, 1816, 1834, 1836, 1849, 1927, 1931, 1986, 1987, 1996, 2021, 2067, 2110, 2121,
                    2122, 2183, 2229, 2472],
                auto1_atomic_sets[2]
            );
            assert_eq!(
                vec![22, 23, 25, 26, 74, 79, 152, 180, 182, 187, 203, 214, 218, 231, 237, 251, 257, 286,
                    298, 300, 349, 404, 410, 463, 492, 592, 652, 661, 680, 702, 717, 760, 770, 808, 848,
                    863, 912, 1002, 1028, 1161, 1238, 1258, 1304, 1446, 1473, 1488, 1500, 1532, 1584, 1603,
                    1630, 1666, 1727, 1739, 1757, 1806, 1890, 2007, 2011, 2017, 2025, 2051, 2086, 2087,
                    2090, 2136, 2200, 2275, 2277, 2280, 2300, 2308, 2336, 2338, 2343, 2483],
                auto1_atomic_sets[3]
            );
            assert_eq!(
                vec![33, 54, 75, 97, 118, 164, 284, 308, 319, 351, 633, 642, 1558, 2010, 2154, 2169, 2193],
                auto1_atomic_sets[4]
            );
        }
    }

    #[test]
    fn empty_candidates() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        assert!(vp9.get_atomic_sets(Some(vec![]), &vec![]).is_empty());
        assert!(auto1.get_atomic_sets(Some(vec![]), &vec![]).is_empty());
    }

    #[test]
    fn candidates_and_assumptions_for_core() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        let vp9_default_as = vp9.get_atomic_sets(None, &vec![]);
        let vp9_core = vp9.core.clone().into_iter().collect_vec();
        assert_eq!(
            vp9_default_as,
            vp9.get_atomic_sets(Some((1..=vp9.number_of_variables as u32).collect_vec()), &vec![])
        );
        assert_eq!(
            vp9_default_as,
            vp9.get_atomic_sets(None, &vp9_core)
        );
        assert_eq!(
            vp9_default_as,
            vp9.get_atomic_sets(Some((1..=vp9.number_of_variables as u32).collect_vec()), &vp9_core)
        );
        assert_eq!(
            vp9_default_as,
            vp9.get_atomic_sets(Some(vp9.core.clone().into_iter().map(|f| f as u32).collect_vec()), &vp9_core)
        );
    }

    #[test]
    fn candidates_and_assumptions() {
        let mut auto1: Ddnnf =
        build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
        let assumptions = vec![10, 20, 35];
        let atomic_sets = 
            auto1.get_atomic_sets(Some((1..=50).collect_vec()), &vec![10, 20, 35])
                .iter()
                .map(|subset| subset.iter().map(|&f| f as i32).collect_vec())
                .collect_vec();

        assert_eq!(
            vec![vec![1, 2, 3, 4, 5, 8, 12, 14, 15, 16, 17, 18, 32, 34, 36, 37, 39, 40, 41, 42, 43], vec![10, 11, 20, 22, 23, 25, 26, 33, 35], vec![19, 44]],
            atomic_sets
        );

        // atomic set property holds for all possibilities
        for subset in atomic_sets.iter() {
            let mut compare_value = Integer::from(-1);
            for feature in subset.iter() {
                let mut query_slice = assumptions.clone();
                query_slice.push(*feature);

                if compare_value == -1 {
                    compare_value = auto1.execute_query(&query_slice);
                } else {
                    assert_eq!(compare_value, auto1.execute_query(&query_slice));
                }
            }
        }

        // atomic sets are supposed to be distinct
        assert_ne!(auto1.execute_query(&atomic_sets[0]), auto1.execute_query(&atomic_sets[1]));
        assert_ne!(auto1.execute_query(&atomic_sets[0]), auto1.execute_query(&atomic_sets[2]));
        assert_ne!(auto1.execute_query(&atomic_sets[1]), auto1.execute_query(&atomic_sets[2]));
    }
}