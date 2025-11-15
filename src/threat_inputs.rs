/* A large part of this file is taken or adapted from montytrain (https://github.com/official-monty/montytrain) */

use bullet::game::inputs::SparseInputType;
use bulletformat::ChessBoard;

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

macro_rules! init_add_assign {
    (|$sq:ident, $init:expr, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size + 1];
        let mut val = $init;
        while $sq < $size {
            res[$sq] = val;
            val += {$($rest)+};
            $sq += 1;
        }

        res[$size] = val;

        res
    }};
}

pub struct Side;
impl Side {
    pub const WHITE: usize = 0;
    pub const BLACK: usize = 1;
}

pub struct Piece;
impl Piece {
    pub const PAWN: usize = 2;
    pub const KNIGHT: usize = 3;
    pub const BISHOP: usize = 4;
    pub const ROOK: usize = 5;
    pub const QUEEN: usize = 6;
    pub const KING: usize = 7;
}

pub mod offsets {
    use super::indices;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = PAWN + 6 * indices::PAWN; // pawn attacks: 6 different pieces: own pawn, own knight, own rook, opp pawn, opp knight, opp rook
    pub const BISHOP: usize = KNIGHT + 12 * indices::KNIGHT[64]; // knight attacks: all 12 pieces possible
    pub const ROOK: usize = BISHOP + 10 * indices::BISHOP[64]; // bishop attacks: queens are skipped
    pub const QUEEN: usize = ROOK + 10 * indices::ROOK[64]; // rook attacks: queens are skipped
    pub const KING: usize = QUEEN + 12 * indices::QUEEN[64]; // queen attacks: all 12 pieces possible
    pub const END: usize = KING + 8 * indices::KING[64]; // king attacks: queen and king skipped
}

pub mod indices {
    use super::attacks;

    pub const PAWN: usize = 84;
    pub const KNIGHT: [usize; 65] =
        init_add_assign!(|sq, 0, 64| attacks::KNIGHT[sq].count_ones() as usize); // for every square (including index 64, aka total) stores the start index of threats of that square
    pub const BISHOP: [usize; 65] =
        init_add_assign!(|sq, 0, 64| attacks::BISHOP[sq].count_ones() as usize);
    pub const ROOK: [usize; 65] =
        init_add_assign!(|sq, 0, 64| attacks::ROOK[sq].count_ones() as usize);
    pub const QUEEN: [usize; 65] =
        init_add_assign!(|sq, 0, 64| attacks::QUEEN[sq].count_ones() as usize);
    pub const KING: [usize; 65] =
        init_add_assign!(|sq, 0, 64| attacks::KING[sq].count_ones() as usize);
}

pub mod attacks {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = A << 7;

    const DIAGS: [u64; 15] = [
        0x0100_0000_0000_0000,
        0x0201_0000_0000_0000,
        0x0402_0100_0000_0000,
        0x0804_0201_0000_0000,
        0x1008_0402_0100_0000,
        0x2010_0804_0201_0000,
        0x4020_1008_0402_0100,
        0x8040_2010_0804_0201,
        0x0080_4020_1008_0402,
        0x0000_8040_2010_0804,
        0x0000_0080_4020_1008,
        0x0000_0000_8040_2010,
        0x0000_0000_0080_4020,
        0x0000_0000_0000_8040,
        0x0000_0000_0000_0080,
    ];

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);

    pub const KING: [u64; 64] = init!(|sq, 64| {
        let mut k = 1 << sq;
        k |= (k << 8) | (k >> 8);
        k |= ((k & !A) >> 1) | ((k & !H) << 1);
        k ^ (1 << sq)
    });
}

pub struct Attacks;
impl Attacks {
    #[inline]
    pub fn pawn(sq: usize, side: usize) -> u64 {
        LOOKUP.pawn[side][sq]
    }

    #[inline]
    pub fn knight(sq: usize) -> u64 {
        LOOKUP.knight[sq]
    }

    #[inline]
    pub fn king(sq: usize) -> u64 {
        LOOKUP.king[sq]
    }

    // hyperbola quintessence
    // this gets automatically vectorised when targeting avx or better
    #[inline]
    pub fn bishop(sq: usize, occ: u64) -> u64 {
        let mask = LOOKUP.bishop[sq];

        let mut diag = occ & mask.diag;
        let mut rev1 = diag.swap_bytes();
        diag = diag.wrapping_sub(mask.bit);
        rev1 = rev1.wrapping_sub(mask.swap);
        diag ^= rev1.swap_bytes();
        diag &= mask.diag;

        let mut anti = occ & mask.anti;
        let mut rev2 = anti.swap_bytes();
        anti = anti.wrapping_sub(mask.bit);
        rev2 = rev2.wrapping_sub(mask.swap);
        anti ^= rev2.swap_bytes();
        anti &= mask.anti;

        diag | anti
    }

    // shifted lookup
    // files and ranks are mapped to 1st rank and looked up by occupancy
    #[inline]
    pub fn rook(sq: usize, occ: u64) -> u64 {
        let flip = ((occ >> (sq & 7)) & File::A).wrapping_mul(DIAG);
        let file_sq = (flip >> 57) & 0x3F;
        let files = LOOKUP.file[sq][file_sq as usize];

        let rank_sq = (occ >> RANK_SHIFT[sq]) & 0x3F;
        let ranks = LOOKUP.rank[sq][rank_sq as usize];

        ranks | files
    }

    #[inline]
    pub fn queen(sq: usize, occ: u64) -> u64 {
        Self::bishop(sq, occ) | Self::rook(sq, occ)
    }
}

struct File;
impl File {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = Self::A << 7;
}

const EAST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) ^ (1 << sq) ^ WEST[sq]);
const WEST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) & ((1 << sq) - 1));
const DIAG: u64 = DIAGS[7];
const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

// masks for hyperbola quintessence bishop attacks
#[derive(Clone, Copy)]
struct Mask {
    bit: u64,
    diag: u64,
    anti: u64,
    swap: u64,
}

struct Lookup {
    pawn: [[u64; 64]; 2],
    knight: [u64; 64],
    king: [u64; 64],
    bishop: [Mask; 64],
    rank: [[u64; 64]; 64],
    file: [[u64; 64]; 64],
}

static LOOKUP: Lookup = Lookup {
    pawn: PAWN,
    knight: KNIGHT,
    king: KING,
    bishop: BISHOP,
    rank: RANK,
    file: FILE,
};

const PAWN: [[u64; 64]; 2] = [
    init!(|sq, 64| (((1 << sq) & !File::A) << 7) | (((1 << sq) & !File::H) << 9)),
    init!(|sq, 64| (((1 << sq) & !File::A) >> 9) | (((1 << sq) & !File::H) >> 7)),
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !File::A) >> 1) | ((k & !File::H) << 1);
    k ^ (1 << sq)
});

const BISHOP: [Mask; 64] = init!(|sq, 64|
    let bit = 1 << sq;
    let file = sq & 7;
    let rank = sq / 8;
    Mask {
        bit,
        diag: bit ^ DIAGS[7 + file - rank],
        anti: bit ^ DIAGS[    file + rank].swap_bytes(),
        swap: bit.swap_bytes()
    }
);

const RANK_SHIFT: [usize; 64] = init!(|sq, 64| sq - (sq & 7) + 1);

const RANK: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| {
    let file = sq & 7;
    let mask = (occ << 1) as u64;
    let east = ((EAST[file] & mask) | (1 << 63)).trailing_zeros() as usize;
    let west = ((WEST[file] & mask) | 1).leading_zeros() as usize ^ 63;
    (EAST[file] ^ EAST[east] | WEST[file] ^ WEST[west]) << (sq - file)
}));

const FILE: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| (RANK[7 - sq / 8][occ]
    .wrapping_mul(DIAG)
    & File::H)
    >> (7 - (sq & 7))));

pub fn map_piece_threat(
    // maps a threat to a feature index (still need to add half width in case of nstm)
    piece: usize,  // piece type
    src: usize,    // square of piece type
    dest: usize,   // square of interaction
    target: usize, // piece (including color) on square of interaction
    enemy: bool,   // indicates whether "target" is an enemy piece (threat or protection)
) -> Option<usize> {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest, target, enemy),
        Piece::KNIGHT => map_knight_threat(src, dest, target),
        Piece::BISHOP => map_bishop_threat(src, dest, target),
        Piece::ROOK => map_rook_threat(src, dest, target),
        Piece::QUEEN => map_queen_threat(src, dest, target),
        Piece::KING => map_king_threat(src, dest, target),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

const fn offset_mapping<const N: usize>(a: [usize; N]) -> [usize; 12] {
    let mut res = [usize::MAX; 12];

    let mut i = 0;
    while i < N {
        res[a[i] - 2] = i; // PieceType - 2 gives the STM index (0..<6)
        res[a[i] + 4] = i + N; // PieceType + 4 gives the OPP index (6..<12)
        i += 1;
    }

    res
}

fn target_is(target: usize, piece: usize) -> bool {
    target % 6 == piece - 2
}

fn map_pawn_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    // target is still a colored piece
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::ROOK]);
    // this MAP call results in the following array:
    // [0, 1, MAX, 2, MAX, MAX, 3, 4, MAX, 5, MAX, MAX]

    // pawn <-> bishop threats are covered by the bishop attacking the pawn, same for queen
    // for pawn <-> pawn threats, we don't cover cases where dest > src to avoid duplicates
    // for pawn <-> pawn threats of the same color, there are never duplicates
    if MAP[target] == usize::MAX || (enemy && dest > src && target_is(target, Piece::PAWN)) {
        None
    } else {
        let up = usize::from(dest > src);
        let diff = dest.abs_diff(src);
        let id = if diff == [9, 7][up] { 0 } else { 1 };
        let attack = 2 * (src % 8) + id - 1;
        let threat = offsets::PAWN + MAP[target] * indices::PAWN + (src / 8 - 1) * 14 + attack;

        assert!(threat < offsets::KNIGHT, "{threat}");

        Some(threat)
    }
}

fn map_knight_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    // don't duplicate knight threats, only allow where dest < src
    if dest > src && target_is(target, Piece::KNIGHT) {
        None
    } else {
        let idx = indices::KNIGHT[src] + below(src, dest, &attacks::KNIGHT);
        let threat = offsets::KNIGHT + target * indices::KNIGHT[64] + idx;

        assert!(threat >= offsets::KNIGHT, "{threat}");
        assert!(threat < offsets::BISHOP, "{threat}");

        Some(threat)
    }
}

fn map_bishop_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([
        Piece::PAWN,
        Piece::KNIGHT,
        Piece::BISHOP,
        Piece::ROOK,
        Piece::KING,
    ]);
    // queen attacks bishop => bishop attacks queen, so ont used here
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::BISHOP) {
        None
    } else {
        let idx = indices::BISHOP[src] + below(src, dest, &attacks::BISHOP);
        let threat = offsets::BISHOP + MAP[target] * indices::BISHOP[64] + idx;

        assert!(threat >= offsets::BISHOP, "{threat}");
        assert!(threat < offsets::ROOK, "{threat}");

        Some(threat)
    }
}

fn map_rook_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([
        Piece::PAWN,
        Piece::KNIGHT,
        Piece::BISHOP,
        Piece::ROOK,
        Piece::KING,
    ]);
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::ROOK) {
        None
    } else {
        let idx = indices::ROOK[src] + below(src, dest, &attacks::ROOK);
        let threat = offsets::ROOK + MAP[target] * indices::ROOK[64] + idx;

        assert!(threat >= offsets::ROOK, "{threat}");
        assert!(threat < offsets::QUEEN, "{threat}");

        Some(threat)
    }
}

fn map_queen_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    if dest > src && target_is(target, Piece::QUEEN) {
        None
    } else {
        let idx = indices::QUEEN[src] + below(src, dest, &attacks::QUEEN);
        let threat = offsets::QUEEN + target * indices::QUEEN[64] + idx;

        assert!(threat >= offsets::QUEEN, "{threat}");
        assert!(threat < offsets::KING, "{threat}");

        Some(threat)
    }
}

fn map_king_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] =
        offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);
    if MAP[target] == usize::MAX {
        None
    } else {
        let idx = indices::KING[src] + below(src, dest, &attacks::KING);
        // piece offset + attacking piece offset + local index (made up of square offset + square-local index)
        let threat = offsets::KING + MAP[target] * indices::KING[64] + idx;

        assert!(threat >= offsets::KING, "{threat}");
        assert!(threat < offsets::END, "{threat}");

        Some(threat)
    }
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

fn map_features_bucketed<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror (sf-like)
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 <= 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(piece, sq, dest, pieces[dest], enemy) {
                        f(side_offset + idx);
                    }
                });
            });
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputsBucketsMirrored;
impl ThreatInputsBucketsMirrored {
    #[rustfmt::skip]
    pub const BUCKETS: [usize; 64] = [
        28, 29, 30, 31, 31, 30, 29, 28,
        24, 25, 26, 27, 27, 26, 25, 24,
        20, 21, 22, 23, 23, 22, 21, 20,
        16, 17, 18, 19, 19, 18, 17, 16,
        12, 13, 14, 15, 15, 14, 13, 12,
        08, 09, 10, 11, 11, 10, 09, 08,
        04, 05, 06, 07, 07, 06, 05, 04,
        00, 01, 02, 03, 03, 02, 01, 00,
    ];
    pub const MK_SIZE: usize = 704;
    pub const BUCKET_COUNT: usize = 32;
    pub const FACTORISER_SIZE: usize = 768;
    pub const THREATS_SIZE: usize = 2 * offsets::END;
    pub const HALFKA_V2_SIZE: usize = 704 * Self::BUCKET_COUNT;

    pub const THREATS_MAX_ACTIVE: usize = 128; // includes PSQ
    pub const FACTORISER_MAX_ACTIVE: usize = 32;

    pub fn make_index(perspective: usize, sq: usize, pc: u8, ksq: usize) -> usize {
        let flip = 56 * perspective;
        let orientation = if ksq % 8 > 3 { 0 } else { 7 };
        let color = usize::from(pc & 8 > 0);
        let pctype = usize::from(pc & 7);
        let sf_pc: usize = 2 * pctype + color ^ perspective;
        let sf_pc = sf_pc.min(10); // no king feature on index 11
        return (sq ^ flip ^ orientation) + 64 * sf_pc + 64 * 11 * Self::BUCKETS[ksq];
    }

    /// This function should only ever be called with HalfKAv2-related indices
    pub fn derive_factorised_feature(feat: usize) -> usize {
        let mut feature = feat % 704;

        let square = feat % 64;
        let bucket = feat / 704;
        let piece = feature / 64;
        if piece == 10
            && (square % 8 <= 3 || ThreatInputsBucketsMirrored::BUCKETS[square] != bucket)
        {
            // If the feature corresponds to a king,
            // and the king is in a different bucket or in an impossible
            // position due to mirroring, it's the ntm king,
            // which is encoded differently in Chess768 inputs
            feature += 64; // effectively makes "piece" become 11
        }

        feature
    }
}
impl SparseInputType for ThreatInputsBucketsMirrored {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        Self::FACTORISER_SIZE + Self::THREATS_SIZE + Self::HALFKA_V2_SIZE
    }

    fn max_active(&self) -> usize {
        Self::THREATS_MAX_ACTIVE + Self::FACTORISER_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let stm = Self::make_index(0 as usize, square as usize, piece, pos.our_ksq() as usize);
            let ntm = Self::make_index(1 as usize, square as usize, piece, pos.opp_ksq() as usize);
            // bucketed feature
            f(Self::FACTORISER_SIZE + stm, Self::FACTORISER_SIZE + ntm);
            // factorised feature
            f(
                Self::derive_factorised_feature(stm),
                Self::derive_factorised_feature(ntm),
            );
        }

        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; Self::THREATS_MAX_ACTIVE];
        map_features_bucketed(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; Self::THREATS_MAX_ACTIVE];
        map_features_bucketed(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            // threat feature
            f(
                Self::FACTORISER_SIZE + Self::HALFKA_V2_SIZE + stm,
                Self::FACTORISER_SIZE + Self::HALFKA_V2_SIZE + ntm,
            );
        }
    }

    fn shorthand(&self) -> String {
        format!(
            "{}x{}+{}",
            Self::THREATS_SIZE,
            Self::MK_SIZE,
            Self::BUCKET_COUNT
        )
    }

    fn description(&self) -> String {
        "Factorised HalfKAv2 + Threat inputs".to_string()
    }
}
