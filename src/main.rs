use bullet::{
    game::{formats::sfbinpack::{TrainingDataEntry, chess::{r#move::MoveType, piecetype::PieceType}}, inputs::{self, Chess768, Factorised, Factorises, SparseInputType}, outputs::{self, OutputBuckets}}, nn::{
        InitSettings, Shape, optimiser::Ranger
    }, trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    }, value::{ValueTrainerBuilder, loader}
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
        "".to_string()
    }

    fn is_factorised(&self) -> bool {
        false
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        
        let make_index = |perspective, sq, pc, ksq| -> usize {
            let flip = 56 * perspective;
            let orientation = if ksq % 8 > 3 { 0 } else { 7 };
            let color = usize::from(pc & 8 > 0);
            let pctype = usize::from(pc & 7);
            let sf_pc: usize = 2 * pctype + color ^ perspective;
            let sf_pc = sf_pc.min(10); // no king feature on index 11
            return (sq ^ flip ^ orientation) + 64 * sf_pc + 64 * 11 * SfInputs::BUCKETS[ksq];
        };

        for (piece, square) in pos.into_iter() {
            let stm = make_index(0 as usize, square as usize, piece, pos.our_ksq() as usize);
            let ntm = make_index(1 as usize, square as usize, piece, pos.opp_ksq() as usize);
            f(stm, ntm);
        }
    }

    fn max_active(&self) -> usize {
        32
    }

    fn num_inputs(&self) -> usize {
        704 * 32
    }

    fn shorthand(&self) -> String {
        "".to_string()
    }

}

impl Factorises<SfInputs> for Chess768 {
    fn derive_feature(&self, _: &SfInputs, feat: usize) -> Option<usize> {
        let mut feature = feat % 704;

        let square = feat % 64;
        let bucket = feat / 704;
        let piece = feature / 64;
        if piece == 10 && (square % 8 <= 3 || SfInputs::BUCKETS[square] != bucket) {
            // If the feature corresponds to a king,
            // and the king is in a different bucket or in an impossible
            // position due to mirroring, it's the ntm king,
            // which is encoded differently in Chess768 inputs
            feature += 64; // effectively makes "piece" become 11
        }

        Some(feature)
    }
}

type InputFeatures = Factorised<SfInputs, Chess768>;
const L1: usize = 3072;
const L2: usize = 15;
const L3: usize = 32;

fn main() {
    let inputs = InputFeatures::from_parts(SfInputs::default(), Chess768::default());

    let output_buckets = SfMaterialCount::default();
    let num_inputs = <InputFeatures as inputs::SparseInputType>::num_inputs(&inputs);
    const NUM_OUTPUT_BUCKETS: usize = <SfMaterialCount as outputs::OutputBuckets<_>>::BUCKETS;

    let saved_format = vec![
        SavedFormat::id("l0b").round().quantise::<i16>(127),
        SavedFormat::id("l0w").transform(move |_, weights| inputs.merge_factoriser(weights)).round().quantise::<i16>(127),
        SavedFormat::id("pst").transform(move |_, weights| inputs.merge_factoriser(weights)).round().quantise::<i32>(600 * 16),
        SavedFormat::id("l1b").round().quantise::<i32>(64 * 127),/*.transform(|store, weights| {
            let fact = store.get("l1_factb").values.repeat(NUM_OUTPUT_BUCKETS);
            weights.into_iter().zip(fact).map(|(a, b)| a + b).collect()
        }),*/
        SavedFormat::id("l1w").round().quantise::<i8>(64).transpose(),/*.transform(|store, weights| {
            let fact = store.get("l1_factw").values.repeat(NUM_OUTPUT_BUCKETS);
            weights.into_iter().zip(fact).map(|(a, b)| a + b).collect()
        }),*/
        SavedFormat::id("l2b").round().quantise::<i32>(64 * 127),
        SavedFormat::id("l2w").round().quantise::<i8>(64).transpose(),
        SavedFormat::id("l3b").round().quantise::<i32>(16 * 600),
        SavedFormat::id("l3w").round().quantise::<i8>(600 * 16 / 127).transpose(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Ranger)
        .loss_fn(|output, targets| output.sigmoid().power_error(targets, 2.6))
        .inputs(inputs)
        .output_buckets(output_buckets)
        .save_format(saved_format.as_slice())
        .build(|builder, stm, ntm, buckets| {
            // trainable weights
            let l0 = builder.new_affine("l0", num_inputs, L1);
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * (L2 + 1));
            // let l1_fact = builder.new_affine("l1_fact", L1, L2 + 1);
            let l2 = builder.new_affine("l2", L2 * 2, NUM_OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, NUM_OUTPUT_BUCKETS);
            let pst = builder.new_weights(
                "pst",
                Shape::new(NUM_OUTPUT_BUCKETS, num_inputs),
                InitSettings::Zeroed,
            );

            // inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            out = l1.forward(out).select(buckets);// + l1_fact.forward(out);

            let skip_neuron = out.slice_rows(15, 16);
            out = out.slice_rows(0, 15);

            out = out.abs_pow(2.0).concat(out);
            out = out.crelu();

            out = l2.forward(out).select(buckets).crelu();
            out = l3.forward(out).select(buckets);

            let stm_pst = pst.matmul(stm).select(buckets);
            let ntm_pst = pst.matmul(ntm).select(buckets);
            let pst_out = stm_pst.linear_comb(0.5, ntm_pst, -0.5);
            out = out + skip_neuron + pst_out;

            out
        });

    println!("Params: {}", trainer.optimiser.graph.get_num_params());

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 600.0,
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
        let file_path = "/mnt/d/Chess Data/aprilmay2022/T79-apr2022-12tb7p.binpack";
        let buffer_size_mb = 1024;
        let threads = 8;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }

        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };
    //trainer.profile_all_nodes();
    trainer.run(&schedule, &settings, &data_loader);
    // trainer.load_from_checkpoint("checkpoints/test-1");
    //trainer.report_profiles();
    let eval = 600.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
    let eval = 600.0 * trainer.eval("r1bq1rk1/ppppbppp/3n4/4R3/8/8/PPPP1PPP/RNBQ1BK1 w - - 1 9 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}