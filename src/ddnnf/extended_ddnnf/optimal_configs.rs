use crate::ddnnf::anomalies::t_wise_sampling::data_structure::Config;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use crate::NodeType::*;

impl ExtendedDdnnf {

    pub fn get_config_objective_fn_val(&self, config: &Config) -> f64 {
        config.get_decided_literals()
            .map(|literal| {
                if literal < 0 { 0.0 } // deselected
                else { self.get_objective_fn_val(literal.unsigned_abs()) }
            })
            .sum()
    }

    //############## Recursive ###################

    pub fn calc_best_config_recursive(&self) -> Option<Config> {
        self.calc_best_config_for_node_recursive(self.ddnnf.nodes.len() - 1)
    }


    pub fn calc_best_config_for_node_recursive(&self, node_id: usize) -> Option<Config> {
        let number_of_variables = self.ddnnf.number_of_variables as usize;
        let node = self.ddnnf.nodes.get(node_id).unwrap_or_else(|| panic!("Node {node_id} does not exist."));

        match &node.ntype {
            True => Some(Config::from(&[], number_of_variables)),
            False => None,
            Literal { literal } => Some(Config::from(&[*literal], number_of_variables)),
            And { children } => {
                let configs = children.iter()
                    .map(|&child_node_id| self.calc_best_config_for_node_recursive(child_node_id));
                let mut unified_config = Config::from(&[], number_of_variables);

                for config_opt in configs {
                    match config_opt {
                        None => return None,
                        Some(config) => unified_config.extend(config.get_decided_literals())
                    }
                }

                Some(unified_config)
            },
            Or { children } => {
                let best_config = children.iter()
                    .map(|&child_node_id| self.calc_best_config_for_node_recursive(child_node_id))
                    .flatten()
                    .max_by(|left, right| { // as Ord is not implemented for f64
                        let left_val = self.get_config_objective_fn_val(left);
                        let right_val = self.get_config_objective_fn_val(right);
                        left_val.total_cmp(&right_val)
                    });
                best_config
            },
        }
    }

    //############## Iterative ###################

    pub fn calc_best_config_iterative(&self) -> Option<Config> {
        let root_node_id = self.ddnnf.nodes.len() - 1;
        self.calc_best_config_for_node_iterative(root_node_id)
    }


    pub fn calc_best_config_for_node_iterative(&self, node_id: usize) -> Option<Config> {
        let node_count = self.ddnnf.nodes.len();
        let mut partial_configs = Vec::with_capacity(node_count);

        for id in 0..=node_id {
            partial_configs.push(self.calc_best_config_for_node_helper_iterative(id, &partial_configs));
        }

        partial_configs.remove(node_id)
    }


    pub fn calc_best_config_for_node_helper_iterative(&self, node_id: usize, partial_configs: &Vec<Option<Config>>) -> Option<Config> {
        let number_of_variables = self.ddnnf.number_of_variables as usize;
        let node = self.ddnnf.nodes.get(node_id).unwrap_or_else(|| panic!("Node {node_id} does not exist."));

        match &node.ntype {
            True => Some(Config::from(&[], number_of_variables)),
            False => None,
            Literal { literal } => Some(Config::from(&[*literal], number_of_variables)),
            And { children } => {
                let configs = children.iter()
                    .map(|&child_node_id| {
                        partial_configs.get(child_node_id).unwrap_or_else(|| panic!("No partial config for node {child_node_id} present."))
                    });
                let mut unified_config = Config::from(&[], number_of_variables);

                for config_opt in configs {
                    match config_opt {
                        None => return None,
                        Some(config) => unified_config.extend(config.get_decided_literals())
                    }
                }

                Some(unified_config)
            },
            Or { children } => {
                let best_config = children.iter()
                    .map(|&child_node_id| {
                        partial_configs.get(child_node_id).unwrap_or_else(|| panic!("No partial config for node {child_node_id} present."))
                    })
                    .flatten()
                    .max_by(|left, right| { // as Ord is not implemented for f64
                        let left_val = self.get_config_objective_fn_val(left);
                        let right_val = self.get_config_objective_fn_val(right);
                        left_val.total_cmp(&right_val)
                    })
                    .cloned();
                best_config
            },
        }
    }
}


#[cfg(test)]
mod test {
    use crate::ddnnf::extended_ddnnf::test::build_sandwich_ext_ddnnf;
    use super::*;

    pub fn build_sandwich_ext_ddnnf_with_objective_function_values() -> ExtendedDdnnf {
        let mut ext_ddnnf = build_sandwich_ext_ddnnf();
        ext_ddnnf.objective_fn_vals = Some(vec![
            9.0,    // Sandwich
            7.0,    // Bread
            1.0,    // Full Grain
            3.0,    // Flatbread
            2.0,    // Toast
            1.0,    // Cheese
            2.0,    // Gouda
            -1.0,   // Sprinkled
            3.0,    // Slice
            4.0,    // Cheddar
            -2.0,   // Cream Cheese
            -2.0,   // Meat
            5.0,    // Salami
            -2.0,   // Ham
            -1.0,   // Chicken Breast
            7.0,    // Vegetables
            2.0,    // Cucumber
            3.0,    // Tomatoes
            -4.0    // Lettuce
        ]);

        ext_ddnnf
    }

    #[test]
    fn finding_complete_optimal_config() {
        let ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();
        let expected_config = Config::from(
            &vec![
                1,    // Sandwich
                2,    // Bread
                -3,   // Full Grain
                4,    // Flatbread
                -5,   // Toast
                6,    // Cheese
                7,    // Gouda
                -8,   // Sprinkled
                9,    // Slice
                10,   // Cheddar
                -11,  // Cream Cheese
                12,   // Meat
                13,   // Salami
                -14,  // Ham
                -15,  // Chicken Breast
                16,   // Vegetables
                17,   // Cucumber
                18,   // Tomatoes
                -19   // Lettuce
            ],
            ext_ddnnf.ddnnf.number_of_variables as usize
        );

        assert_eq!(ext_ddnnf.calc_best_config_iterative(), Some(expected_config.clone()));
        assert_eq!(ext_ddnnf.calc_best_config_recursive(), Some(expected_config));
    }
}