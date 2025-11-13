use crate::outputs::OutputBuckets;
use bullet::{
    NetworkTrainer, default::inputs::{Chess768, Factorised, SparseInputType}, nn::{
        Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, Node, Shape, optimiser::{RangerOptimiser, RangerParams}
    }, trainer::{
        default::{
            Trainer, formats::sfbinpack::{
                TrainingDataEntry, chess::{r#move::MoveType, piecetype::PieceType}
            }, inputs, loader, outputs
        },
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    }
};
use bulletformat::ChessBoard;

#[derive(Clone, Copy, Default)]
pub struct SfMaterialCount;
impl OutputBuckets<ChessBoard> for SfMaterialCount {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let piece_count = pos.occ().count_ones() as u8 - 1;
        (piece_count / 4) as u8
    }
}

#[derive(Clone, Copy, Default)]
pub struct SfInputs;
impl SfInputs {
    #[rustfmt::skip]
    const BUCKETS: [usize; 64] = [
        28, 29, 30, 31, 31, 30, 29, 28,
        24, 25, 26, 27, 27, 26, 25, 24,
        20, 21, 22, 23, 23, 22, 21, 20,
        16, 17, 18, 19, 19, 18, 17, 16,
        12, 13, 14, 15, 15, 14, 13, 12,
        08, 09, 10, 11, 11, 10, 09, 08,
        04, 05, 06, 07, 07, 06, 05, 04,
        00, 01, 02, 03, 03, 02, 01, 00,
    ];
}

impl SparseInputType for SfInputs {
    type RequiredDataType = ChessBoard;

    fn description(&self) -> String {
        ""
    }

    fn is_factorised(&self) -> bool {
        false
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        
        let make_index = |perspective, sq, pc, ksq| -> usize {
            let flip = 56 * perspective;
            let orientation = if ksq % 8 > 3 { 0 } else { 7 };
            let color = usize::from(piece & 8 > 0);
            let pctype = usize::from(pc & 7);
            let sf_pc = 2 * pctype + (1 - color).max(if pctype == 5 { 0 } else { 1 });
            assert!(pctype <= 5);
            assert!(sf_pc <= 11);
            return (sq ^ flip ^ orientation) + 64 * sf_pc + 64 * 11 * SfInputs::BUCKETS[ksq ^ flip];
        };

        for (piece, square) in pos.into_iter() {
            let stm = make_index(0, square, piece, pos.our_ksq());
            let ntm = make_index(1, square, piece, pos.opp_ksq());
            f(stm, ntm);
        }
    }

    fn max_active(&self) -> usize {
        32
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        
    }

    fn num_inputs(&self) -> usize {
        704 * 32
    }

    fn shorthand(&self) -> String {
        ""
    }

}

type InputFeatures = Factorised<SfInputs, Chess768>;
const L1: usize = 3072;
const L2: usize = 15;
const L3: usize = 32;

fn main() {
    let inputs = InputFeatures::default();

    let output_buckets = SfMaterialCount::default();
    let num_inputs = <InputFeatures as inputs::SparseInputType>::num_inputs(&inputs);
    let max_active = <InputFeatures as inputs::SparseInputType>::max_active(&inputs);
    let num_buckets = <SfMaterialCount as outputs::OutputBuckets<_>>::BUCKETS;

    let (graph, output_node) = build_network(num_inputs, max_active, num_buckets);

    let saved_format = vec![
        SavedFormat::new("l0b", QuantTarget::I16(255), Layout::Normal),
        SavedFormat::new("l0w", QuantTarget::I16(255), Layout::Normal),
        SavedFormat::new("pst", QuantTarget::I32(255), Layout::Normal),
        SavedFormat::new("l1b", QuantTarget::I32(64 * 255), Layout::Normal).add_transform(
            |graph, mut weights| {
                let fact = graph.get_weights("l1_factb").get_dense_vals().unwrap();
                add_factoriser(&mut weights, &fact, L2 + 1);
                weights
            },
        ),
        SavedFormat::new(
            "l1w",
            QuantTarget::I8(64),
            Layout::Transposed(Shape::new(NUM_BUCKETS * (L2 + 1), L1)),
        )
        .add_transform(|graph, mut weights| {
            let fact = graph.get_weights("l1_factw").get_dense_vals().unwrap();
            let fact = SavedFormat::transpose(Shape::new(L2 + 1, L1), &fact);
            add_factoriser(&mut weights, &fact, (L2 + 1) * L1);
            weights
        }),
        SavedFormat::new("l2b", QuantTarget::I32(64 * 127), Layout::Normal),
        SavedFormat::new(
            "l2w",
            QuantTarget::I8(64),
            Layout::Transposed(Shape::new(NUM_BUCKETS * L3, L2 * 2)),
        ),
        SavedFormat::new("l3b", QuantTarget::I32(16 * 600), Layout::Normal),
        SavedFormat::new(
            "l3w",
            QuantTarget::I8(600 * 16 / 127),
            Layout::Transposed(Shape::new(NUM_BUCKETS, L3)),
        ),
    ];

    let mut trainer = Trainer::<RangerOptimiser, _, _>::new(
        graph,
        output_node,
        RangerParams::default(),
        inputs,
        output_buckets,
        saved_format,
        false,
    );

    trainer.mark_weights_as_input_factorised(&["l0w", "pst"]);

    println!("Params: {}", trainer.optimiser().graph.get_num_params());

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 1,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.3,
            step: 60,
        },
        save_rate: 150,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 512,
    };

    let data_loader = {
        let file_path = "/mnt/g/stockfish-data/leela96-filt-v2.min.binpack";
        let buffer_size_mb = 1024;
        let threads = 8;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to).piece_type() == PieceType::None
        }

        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };
    //trainer.profile_all_nodes();
    trainer.run(&schedule, &settings, &data_loader);
    //trainer.report_profiles();
    let eval =
        400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}

fn build_network(num_inputs: usize, max_active: usize, num_buckets: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(num_buckets, 1), 1);

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, L1);
    let l1 = builder.new_affine("l1", L1, num_buckets * (L2 + 1));
    let l1_fact = builder.new_affine("l1_fact", L1, L2 + 1);
    let l2 = builder.new_affine("l2", L2 * 2, num_buckets * L3);
    let l3 = builder.new_affine("l3", L3, num_buckets);
    let pst = builder.new_weights(
        "pst",
        Shape::new(num_buckets, num_inputs),
        InitSettings::Zeroed,
    );

    // inference
    let stm_subnet = l0.forward(stm).activate(Activation::CReLU);
    let ntm_subnet = l0.forward(nstm).activate(Activation::CReLU);
    let mut out = stm_subnet.concat(ntm_subnet);

    out = out.pairwise_mul_post_affine_dual();
    out = l1.forward(out).select(buckets) + l1_fact.forward(out);

    let skip_neuron = out.slice_rows(15, 16);
    out = out.slice_rows(0, 15);

    out = out.concat(out.activate(Activation::Square));
    out = out.activate(Activation::CReLU);

    out = l2.forward(out).select(buckets).activate(Activation::SCReLU);
    out = l3.forward(out).select(buckets);

    let stm_pst = pst.matmul(stm).select(buckets);
    let ntm_pst = pst.matmul(nstm).select(buckets);
    let pst_out = stm_pst.linear_comb(0.5, ntm_pst, -0.5);
    out = out + skip_neuron + pst_out;

    let pred = out.activate(Activation::Sigmoid);
    pred.mpe(targets, 2.6);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}

const NUM_BUCKETS: usize = <SfMaterialCount as outputs::OutputBuckets<_>>::BUCKETS;

fn add_factoriser(weights: &mut [f32], factoriser: &[f32], size: usize) {
    for i in 0..NUM_BUCKETS * size {
        weights[i] += factoriser[i % size]
    }
}
