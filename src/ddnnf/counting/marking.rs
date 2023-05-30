use rug::{Integer, Assign, Complete};

use super::super::node::{NodeType::*};
use crate::Ddnnf;

impl Ddnnf {
    #[inline]
    // Computes the cardinality of a node using the marking algorithm:
    // And and Or nodes that base the computation on their child nodes use the
    // .temp value if the child node is marked and the .count value if not
    pub(crate) fn calc_count_marked_node(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                let marked_children = children.iter().filter(|&&child| self.nodes[child].marker).collect::<Vec<&usize>>();
                self.nodes[i].temp = if marked_children.len() <= children.len() / 2 {
                    marked_children.iter().fold(self.nodes[i].count.clone(), |mut acc, &&index| {
                        let node = &self.nodes[index];
                        if node.count != 0 {
                            acc /= &node.count;
                        }
                        acc *= &node.temp;
                        acc
                    })
                } else {
                    Integer::product(children
                        .iter()
                        .map(|&index| {
                            let node = &self.nodes[index];
                            if node.marker {
                                &node.temp
                            } else {
                                &node.count
                            }
                        }))
                        .complete()
                }
            }
            Or { children } => {
                self.nodes[i].temp = Integer::sum(children.iter().map(|&index| {
                    let node = &self.nodes[index];
                    if node.marker {
                        &node.temp
                    } else {
                        &node.count
                    }
                }))
                .complete()
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1) // True and Literal
        }
    }

    #[inline]
    // Computes the cardinality of a feature and partial configurations using the marking algorithm
    fn operate_on_marker(&mut self, indexes: &[usize], operation: fn(&mut Ddnnf, usize)) -> Integer {
        self.mark_assumptions(indexes);

        // calc the count for all marked nodes, respectevly all nodes that matter
        for j in 0..self.md.len() {
            operation(self, self.md[j]);
        }

        // reset everything
        for index in &self.md {
            self.nodes[*index].marker = false;
        }
        for &index in indexes {
            self.nodes[index].marker = false;
        }
        self.md.clear();

        // the result is propagated through the whole graph up to the root
        self.rt()
    }

    #[inline]
    /// Computes the cardinality of a feature using the marking algorithm.
    /// The marking algorithm differs to the standard variation by only reomputing the
    /// marked nodes. Further, the marked nodes use the .temp value of the childs nodes if they
    /// are also marked and the .count value if they are not.
    pub(crate) fn card_of_feature_with_marker(&mut self, feature: i32) -> Integer {
        if self.has_no_effect_on_query(&feature) {
            self.rc()
        } else if self.makes_query_unsat(&feature) {
            Integer::ZERO
        } else {
            match self.literals.get(&-feature).cloned() {
                Some(i) => self.operate_on_marker(&[i], Ddnnf::calc_count_marked_node),
                // there is no literal corresponding to the feature number and because of that we don't have to do anything besides returning the count of the model
                None => self.rc(),
            }
        }
    }

    #[inline]
    /// Computes the cardinality of a partial configuration using the marking algorithm.
    /// Works analog to the card_of_feature_with_marker()
    pub(crate) fn operate_on_partial_config_marker(
        &mut self,
        features: &[i32],
        operation: fn(&mut Ddnnf, usize)
    ) -> Integer {
        if self.query_is_not_sat(features) {
            Integer::ZERO
        } else {
            let features: Vec<i32> = self.reduce_query(features);
            let indexes: Vec<usize> = self.map_features_opposing_indexes(&features);
            if indexes.is_empty() {
                return self.rc();
            }

            self.operate_on_marker(&indexes, operation)
        }
    }

    /// Creates a clone of the nodes which were marked when computing the cardinality
    /// for a given partial configuration.
    pub fn get_marked_nodes_clone(&mut self, features: &[i32]) -> Vec<usize> {           
        let opposing_indices = &self.map_features_opposing_indexes(features);
        self.mark_assumptions(opposing_indices);
        
        // add the literal nodes
        for index in opposing_indices.iter() {
            self.md.push(*index);
        }
        self.md.sort_unstable();
        let marked_nodes = self.md.clone();
        
        // reset everything
        self.md.clear();
        for node in self.nodes.iter_mut() {
            node.marker = false;
        }

        marked_nodes
    }

    #[inline]
    // marks the nodes under the assumptions that all nodes, provided via
    // indexes are literals that are deselected
    fn mark_assumptions(&mut self, indexes: &[usize]) {
        for index in indexes.iter().copied() {
            self.nodes[index].temp.assign(0); // change the value of the node
            self.mark_nodes_start(index); // go through the path til the root node is marked
        }

        // sort the marked nodes so that we make sure to first calculate the childnodes and then their parents
        self.md.sort_unstable();
    }

    #[inline]
    // marks the nodes starting from an initial Literal. All parents and parents of parents til
    // the root nodes get marked
    fn mark_nodes_start(&mut self, i: usize) {
        self.nodes[i].marker = true;

        for parent in self.nodes[i].parents.clone() {
            // check for parent nodes and adjust their count resulting of the changes to their children
            if !self.nodes[parent].marker {
                // only mark those nodes which aren't already marked to specificly avoid marking nodes near the root multple times
                self.mark_nodes(parent);
            }
        }
    }

    #[inline]
    // marks the nodes starting from an initial Literal. All parents and parents of parents til
    // the root nodes get marked
    fn mark_nodes(&mut self, i: usize) {
        self.nodes[i].marker = true;
        self.md.push(i);

        for parent in self.nodes[i].parents.clone() {
            // check for parent nodes and adjust their count resulting of the changes to their children
            if !self.nodes[parent].marker {
                // only mark those nodes which aren't already marked to specificly avoid marking nodes near the root multple times
                self.mark_nodes(parent);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;

    #[test]
    fn marking_nodes() {
        let mut ddnnf = build_ddnnf("tests/data/small_ex_c2d.nnf", None);
        
        ddnnf.mark_assumptions(&[0]);
        assert_eq!(vec![11], ddnnf.md);
        for node in ddnnf.nodes.iter_mut() {
            node.marker = false;
        }
        ddnnf.md.clear();

        ddnnf.mark_assumptions(&[2, 4]);
        assert_eq!(vec![3, 6, 7, 11], ddnnf.md);
        for node in ddnnf.nodes.iter_mut() {
            node.marker = false;
        }
        ddnnf.md.clear();

        assert!(ddnnf.get_marked_nodes_clone(&[]).is_empty());
        assert_eq!(vec![4, 6, 7, 11], ddnnf.get_marked_nodes_clone(&[2]));
        assert_eq!(vec![9, 10, 11], ddnnf.get_marked_nodes_clone(&[4]));
        assert_eq!(vec![4, 6, 7, 9, 10, 11], ddnnf.get_marked_nodes_clone(&[2, 4]));
        assert_eq!(vec![2, 3, 7, 9, 10, 11], ddnnf.get_marked_nodes_clone(&[1, 3, 4]));
    }
}