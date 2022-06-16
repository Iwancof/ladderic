use alloc::vec::Vec;

use core::cell::RefCell;
use core::fmt::Display;
use core::ops::{Add, Mul};

pub trait Network<InputScl, OutputScl>
where
    InputScl: Default,
    OutputScl: Default,
{
    fn output_size(&self) -> usize;

    fn forward_in_place_unchecked(&self, inp: &[InputScl], out: &mut [OutputScl]);
    fn forward_in_place(&self, inp: &[InputScl], out: &mut [OutputScl]) {
        assert_eq!(self.output_size(), out.len());
        self.forward_in_place_unchecked(inp, out);
    }
    fn forward(&self, inp: &[InputScl]) -> Vec<OutputScl> {
        let mut out = (0..self.output_size())
            .map(|_| OutputScl::default())
            .collect::<Vec<_>>();
        self.forward_in_place_unchecked(inp, &mut out);
        out
    }

    // backward implementation here.
}

#[derive(Debug)]
pub struct Neuron<WeightScl, BiasScl> {
    weight: Vec<WeightScl>,
    bias: BiasScl,
}

#[derive(Debug)]
pub struct Layer<ValueScl, WeightScl, BiasScl> {
    params: Vec<Neuron<WeightScl, BiasScl>>,
    afp: fn(ValueScl) -> ValueScl,
}

#[derive(Debug)]
pub struct DenseNetwork<ValueScl, WeightScl, BiasScl>
where
    ValueScl: Mul<WeightScl, Output = ValueScl>,
    ValueScl: Add<BiasScl, Output = ValueScl>,
{
    working: [RefCell<Vec<ValueScl>>; 2],
    input_size: usize,
    // pub layers: Vec<(Vec<(Vec<WeightScl>, BiasScl)>, fn(ValueScl) -> ValueScl)>,
    pub layers: Vec<Layer<ValueScl, WeightScl, BiasScl>>,
}

impl<ValueScl, WeightScl, BiasScl> Display for DenseNetwork<ValueScl, WeightScl, BiasScl>
where
    ValueScl: Mul<WeightScl, Output = ValueScl>,
    ValueScl: Add<BiasScl, Output = ValueScl>,
    ValueScl: Display,
    BiasScl: Display,
    WeightScl: Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for Layer { params, afp } in &self.layers {
            writeln!(f, "{:?}:", afp)?;
            for Neuron { weight, bias } in params {
                write!(f, "  weight: ")?;
                for b in weight {
                    write!(f, "{} ", b)?;
                }
                write!(f, " bias: {}", bias)?;
                writeln!(f, "")?;
            }
        }

        Ok(())
    }
}

impl<ValueScl, BiasScl, WeightScl> DenseNetwork<ValueScl, WeightScl, BiasScl>
where
    ValueScl: Mul<WeightScl, Output = ValueScl>,
    ValueScl: Add<BiasScl, Output = ValueScl>,
    ValueScl: Default,
    BiasScl: Default,
    WeightScl: Default,
{
    pub fn new_with_gen<WeightGenerator, BiasGenerator>(
        input_size: usize,
        count_act: &[(usize, fn(ValueScl) -> ValueScl)],
        gen_weight: WeightGenerator,
        gen_bias: BiasGenerator,
    ) -> Self
    where
        WeightGenerator: Fn(usize, usize, usize) -> WeightScl,
        BiasGenerator: Fn(usize, usize) -> BiasScl,
    {
        let max = count_act.iter().max().unwrap().0;
        let working = [
            RefCell::new((0..max).map(|_| ValueScl::default()).collect()),
            RefCell::new((0..max).map(|_| ValueScl::default()).collect()),
        ];
        let layers = count_act
            .iter()
            .enumerate()
            .fold(
                (Vec::new(), input_size),
                |(mut layers, prev_size), (layer_index, &(size, ac_fp))| {
                    let bias_weight = (0..size)
                        .map(|dest_neuron| Neuron {
                            weight: (0..prev_size)
                                .map(|src_neuron| gen_weight(layer_index, src_neuron, dest_neuron))
                                .collect(),
                            bias: gen_bias(layer_index, dest_neuron),
                        })
                        .collect();

                    layers.push(Layer {
                        params: bias_weight,
                        afp: ac_fp,
                    });
                    (layers, size)
                },
            )
            .0;

        DenseNetwork {
            working,
            input_size,
            layers,
        }
    }

    pub fn new(input_size: usize, count_act: &[(usize, fn(ValueScl) -> ValueScl)]) -> Self {
        DenseNetwork::new_with_gen(
            input_size,
            count_act,
            |_, _, _| WeightScl::default(),
            |_, _| BiasScl::default(),
        )
    }
}

impl<ValueScl, BiasScl, WeightScl> Network<ValueScl, ValueScl>
    for DenseNetwork<ValueScl, WeightScl, BiasScl>
where
    ValueScl: Default + Clone,
    ValueScl: Add<ValueScl, Output = ValueScl>,
    ValueScl: Mul<WeightScl, Output = ValueScl>,
    ValueScl: Add<BiasScl, Output = ValueScl>,
    BiasScl: Clone,
    WeightScl: Clone,
{
    fn output_size(&self) -> usize {
        self.layers.last().unwrap().params.len()
    }

    fn forward_in_place_unchecked(&self, inp: &[ValueScl], out: &mut [ValueScl]) {
        assert_eq!(inp.len(), self.input_size);

        for (buf, inp) in self.working[0].borrow_mut().iter_mut().zip(inp.iter()) {
            *buf = inp.clone();
            // Use buffer[0] as input.
        }

        let mut last_write = 0;
        for (i, Layer { params, afp }) in self.layers.iter().enumerate() {
            let from = self.working[i % 2].borrow_mut(); // Previous layer.
            let mut to = self.working[(i + 1) % 2].borrow_mut(); // Current layer.
            last_write = (i + 1) % 2;

            // TODO: BUG! `to` must be restricted accessing by currett layer neurons number.
            for (to, Neuron { weight, bias }) in to.iter_mut().zip(params.iter()) {
                // to: Destination. wei: Weight. bia: Bias
                *to = ValueScl::default();

                // if ValueScl were primitive, clone can be replaced to copy.
                for (from, wei) in from.iter().zip(weight.iter()) {
                    *to = to.clone() + (from.clone() * wei.clone());
                }
                *to = to.clone() + bias.clone();

                *to = afp(to.clone());
            }
        }

        for (buf, out) in self.working[last_write].borrow_mut().iter_mut().zip(out) {
            *out = buf.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn linear(x: f64) -> f64 {
        x
    }

    #[test]
    fn forward_test() {
        let mut dn = DenseNetwork::<f64, f64, f64>::new(2, &[(3, linear), (2, linear)]);

        dn.layers[0].params[0].weight[0] = 2.0;
        dn.layers[0].params[0].weight[1] = 0.0;
        dn.layers[0].params[1].weight[0] = 1.0;
        dn.layers[0].params[1].weight[1] = 1.0;
        dn.layers[0].params[2].weight[0] = 0.0;
        dn.layers[0].params[2].weight[1] = 2.0;

        dn.layers[0].params[0].bias = 2.0;
        dn.layers[0].params[1].bias = 2.0;
        dn.layers[0].params[2].bias = 2.0;

        dn.layers[1].params[0].weight[0] = 1.0;
        dn.layers[1].params[0].weight[1] = 1.0;
        dn.layers[1].params[0].weight[2] = 1.0;
        dn.layers[1].params[1].weight[0] = 0.0;
        dn.layers[1].params[1].weight[1] = 0.0;
        dn.layers[1].params[1].weight[2] = 1.0;

        dn.layers[1].params[0].bias = 0.0;
        dn.layers[1].params[1].bias = 0.0;

        let ret = dn.forward(&[3., 4.]);

        assert_eq!(ret[0], 27.);
        assert_eq!(ret[1], 10.);
    }

    #[test]
    fn random_network() {
        use rand::Rng;

        let dn = DenseNetwork::<f64, f64, f64>::new_with_gen(
            3,
            &[
                (5, linear),
                (8, linear),
                (4, linear),
                (6, linear),
                (10, linear),
            ],
            |_l, _d, _s| {
                let mut rng = rand::thread_rng();
                rng.gen()
            },
            |_l, _d| {
                let mut rng = rand::thread_rng();
                rng.gen()
            },
        );

        let _ret = dn.forward(&[1., 2., 3.]);
    }

    #[test]
    fn forward_test_2() {
        let mut dn =
            DenseNetwork::<f64, f64, f64>::new(3, &[(5, linear), (1, linear), (5, linear)]);

        dn.layers[0].params[0].weight[0] = 1.;
        dn.layers[1].params[0].weight[0] = 1.;
        dn.layers[2].params[0].weight[0] = 1.;

        let ret = dn.forward(&[1., 2., 3.]);
        assert_eq!(ret, vec![1., 0., 0., 0., 0.]);
    }

    #[test]
    fn forward_test_3() {
        let mut dn = DenseNetwork::<f64, f64, f64>::new(
            3,
            &[(5, linear), (1, linear), (5, linear), (1, linear)],
        );

        dn.layers[0].params[0].weight[0] = 1.;
        dn.layers[1].params[0].weight[0] = 1.;
        dn.layers[2].params[0].weight[0] = 1.;
        dn.layers[3].params[0].weight[0] = 1.;

        let ret = dn.forward(&[1., 2., 3.]);
        assert_eq!(ret, vec![1.]);
    }
}
