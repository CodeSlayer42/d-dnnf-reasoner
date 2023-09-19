mod objective_function;
mod optimal_configs;

use std::collections::HashMap;
use itertools::Itertools;
use crate::Ddnnf;
use crate::ddnnf::extended_ddnnf::Attribute::{BoolAttr, FloatAttr, IntegerAttr, StringAttr};
use crate::ddnnf::extended_ddnnf::AttributeValue::{BoolVal, FloatVal, IntegerVal, StringVal};
use crate::ddnnf::extended_ddnnf::objective_function::ObjectiveFn;


#[derive(Clone, Debug)]
pub struct ExtendedDdnnf {
    ddnnf: Ddnnf,
    attrs: HashMap<String, Attribute>,
    objective_fn_vals: Option<Vec<f64>>
}


#[derive(Clone, Debug)]
pub struct AttributeT<T> {
    vals: Vec<Option<T>>,
    default_val: Option<T>
}


impl <T> AttributeT<T> {
    pub fn get_val(&self, var: u32) -> Option<&T> {
        self.vals[(var-1) as usize].as_ref().or(self.default_val.as_ref())
    }
}


#[derive(Clone, Debug)]
pub enum Attribute {
    IntegerAttr(AttributeT<i32>),
    FloatAttr(AttributeT<f64>),
    BoolAttr(AttributeT<bool>),
    StringAttr(AttributeT<String>)
}


#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    IntegerVal(i32),
    FloatVal(f64),
    BoolVal(bool),
    StringVal(String)
}


impl Attribute {
    pub fn get_val(&self, var: u32) -> Option<AttributeValue> {
        match self {
            IntegerAttr(attr) => attr.get_val(var).map(|val| IntegerVal(*val)),
            FloatAttr(attr) => attr.get_val(var).map(|val| FloatVal(*val)),
            BoolAttr(attr) => attr.get_val(var).map(|val| BoolVal(*val)),
            StringAttr(attr) => attr.get_val(var).map(|val| StringVal(val.clone()))
        }
    }

    pub fn new_integer_attr(values: Vec<Option<i32>>, default_val: Option<i32>) -> Self {
        IntegerAttr(AttributeT{ vals: values, default_val })
    }

    pub fn new_float_attr(values: Vec<Option<f64>>, default_val: Option<f64>) -> Self {
        FloatAttr(AttributeT{ vals: values, default_val })
    }

    pub fn new_bool_attr(values: Vec<Option<bool>>, default_val: Option<bool>) -> Self {
        BoolAttr(AttributeT{ vals: values, default_val })
    }

    pub fn new_string_attr(values: Vec<Option<String>>, default_val: Option<String>) -> Self {
        StringAttr(AttributeT{ vals: values, default_val })
    }
}


impl Default for ExtendedDdnnf {
    fn default() -> Self {
        let ddnnf = Ddnnf::default();
        let attrs = HashMap::new();

        ExtendedDdnnf{ ddnnf, attrs, objective_fn_vals: None }
    }
}


impl ExtendedDdnnf {
    pub fn new(
        ddnnf: Ddnnf,
        attributes: HashMap<String, Attribute>
    ) -> Self {
        ExtendedDdnnf{ ddnnf, attrs: attributes, objective_fn_vals: None }
    }

    pub fn get_attr_val(&self, attr_name: &str, var: u32) -> Option<AttributeValue> {
        if let Some(attr) = self.attrs.get(attr_name) {
            return attr.get_val(var);
        }
        None
    }

    pub fn get_objective_fn_val(&self, var: u32) -> f64 {
        match &self.objective_fn_vals {
            Some(vals) => vals[(var-1) as usize],
            None => panic!("Objective function values have not been calculated.")
        }
    }

    pub fn calc_objective_fn_vals(&mut self, objective_fn: &ObjectiveFn) {
        let vals = (1..=self.ddnnf.number_of_variables)
            .map(|var| objective_fn.eval(var, &self.attrs))
            .collect_vec();
        self.objective_fn_vals = Some(vals);
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::{build_attributes, build_ddnnf};

    pub fn build_sandwich_ext_ddnnf() -> ExtendedDdnnf {
        let ddnnf = build_ddnnf("tests/data/sandwich.nnf", None);
        let attributes = build_attributes("tests/data/sandwich_attribute_vals.csv");
        ExtendedDdnnf::new(ddnnf, attributes)
    }

    #[test]
    fn building_and_reading_attributes() {
        let ext_ddnnf = build_sandwich_ext_ddnnf();

        assert_eq!(ext_ddnnf.get_attr_val("Calories", 3).unwrap(), IntegerVal(203));        // Full Grain
        assert_eq!(ext_ddnnf.get_attr_val("Price", 14).unwrap(), FloatVal(0.99));           // Ham
        assert_eq!(ext_ddnnf.get_attr_val("Organic Food", 11).unwrap(), BoolVal(false));    // Cream Cheese
    }

    #[test]
    fn calculation_of_objective_function_values() {
        use self::objective_function::{ObjectiveFn::*, Condition::*};

        let mut ext_ddnnf = build_sandwich_ext_ddnnf();
        let objective_fn = IfElse(
            And(
                BoolVar("Organic Food".to_string()).boxed(),
                LessThan(NumVar("Calories".to_string()).boxed(), 10.0).boxed()
            ).boxed(),
            Mul(
                NumVar("Calories".to_string()).boxed(),
                NumConst(100.0).boxed()
            ).boxed(),
            Neg(
                NumVar("Price".to_string()).boxed()
            ).boxed()
        );

        let expected = vec![
            0.0,    // Sandwich
            0.0,    // Bread
            -1.99,  // Full Grain
            -1.79,  // Flatbread
            -1.79,  // Toast
            0.0,    // Cheese
            0.0,    // Gouda
            -0.49,  // Sprinkled
            -0.69,  // Slice
            -0.69,  // Cheddar
            -0.59,  // Cream Cheese
            0.0,    // Meat
            -1.29,  // Salami
            -0.99,  // Ham
            -1.39,  // Chicken Breast
            0.0,    // Vegetables
            200.0,  // Cucumber
            300.0,  // Tomatoes
            200.0   // Lettuce
        ];

        ext_ddnnf.calc_objective_fn_vals(&objective_fn);

        assert_eq!(ext_ddnnf.objective_fn_vals.unwrap(), expected);
    }
}


