"""Chess: Naluri vs Laya, each move a typed `choice` question over the legal moves.

Neither model was trained on chess, so this is a fun stress test, not a measure of decision quality.
Every option is a legal move described in plain English; positions with many legal moves use a small
knockout (groups of <= 12, then a final), since both models are built for ~20 options at most.
Games longer than --max-plies are adjudicated on material. PGNs go to results/chess/.
Usage: python chess_battle.py [--games 4] [--max-plies 120]
"""
import argparse
import random
import time
from pathlib import Path

import chess
import chess.pgn

import s1

PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
GROUP = 12


def describe(board, move):
    """Plain-English description of a legal move, so the models get words, not just notation."""
    if board.is_castling(move):
        text = "castle " + ("kingside" if board.is_kingside_castling(move) else "queenside")
    else:
        piece = board.piece_at(move.from_square)
        text = f"{PIECE[piece.piece_type]} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
        if board.is_capture(move):
            cap = board.piece_at(move.to_square)
            text += f", captures a {PIECE[cap.piece_type] if cap else 'pawn'}"
        if move.promotion:
            text += f", promotes to a {PIECE[move.promotion]}"
    if board.gives_check(move):
        b = board.copy(); b.push(move)
        text += ", checkmate" if b.is_checkmate() else ", gives check"
    return text


def state_of(board):
    side = "white" if board.turn else "black"
    return {"side_to_move": side, "fen": board.fen(),
            "board": str(board),
            "material": {"white": material(board, chess.WHITE), "black": material(board, chess.BLACK)},
            "in_check": board.is_check(),
            "recent_moves": " ".join(m for m in recent_san(board, 8))}


def recent_san(board, n):
    b = chess.Board()
    out = []
    for mv in board.move_stack:
        out.append(b.san(mv)); b.push(mv)
    return out[-n:]


def material(board, color):
    return sum(VALUE[p.piece_type] for p in board.piece_map().values() if p.color == color)


class Naluri:
    name = "Naluri (M6-pairs)"

    def __init__(self):
        self.tok, self.model = s1.load("runs/naluri-xlmr-base-m6-pairs_only", device="cuda")

    def scores(self, state, question, options):
        q = {"type": "choice", "instructions": question, "criteria": options}
        return s1.predict(self.tok, self.model, state, {"m": q})["m"]


class Laya:
    name = "Laya (English)"

    def __init__(self):
        import laya
        self.agent = laya.load("convaiinnovations/laya", device="cuda")

    def scores(self, state, question, options):
        q = {"type": "choice", "instructions": question, "criteria": options}
        return self.agent.predict(state, {"m": q})["answers"]["m"]["probabilities"]


def choose(player, board, rng):
    moves = list(board.legal_moves)
    rng.shuffle(moves)  # no fixed order: options are presented in a random order each turn
    san = {board.san(m): m for m in moves}
    desc = {board.san(m): describe(board, m) for m in moves}
    state = state_of(board)
    question = f"Which move is best for {state['side_to_move']} in this chess position?"
    pool = list(san)
    while len(pool) > GROUP:  # knockout: best of each group, then compare the winners
        winners = []
        for i in range(0, len(pool), GROUP):
            group = pool[i:i + GROUP]
            if len(group) == 1:
                winners += group
                continue
            p = player.scores(state, question, {k: desc[k] for k in group})
            winners.append(max(p, key=p.get))
        pool = winners
    if len(pool) == 1:
        return san[pool[0]], 1.0
    p = player.scores(state, question, {k: desc[k] for k in pool})
    best = max(p, key=p.get)
    return san[best], p[best]


def play(white, black, max_plies, rng, game_no):
    board = chess.Board()
    t_white = t_black = 0.0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        player = white if board.turn else black
        t = time.perf_counter()
        move, _ = choose(player, board, rng)
        dt = time.perf_counter() - t
        if board.turn:
            t_white += dt
        else:
            t_black += dt
        board.push(move)
    if board.is_game_over(claim_draw=True):
        result, reason = board.result(claim_draw=True), board.outcome(claim_draw=True).termination.name.lower()
    else:  # adjudicate on material
        mw, mb = material(board, chess.WHITE), material(board, chess.BLACK)
        result = "1-0" if mw > mb + 1 else "0-1" if mb > mw + 1 else "1/2-1/2"
        reason = f"adjudicated on material {mw}-{mb} after {max_plies} plies"
    game = chess.pgn.Game.from_board(board)
    game.headers.update({"Event": "Naluri vs Laya (System One chess)", "Round": str(game_no), "White": white.name,
                         "Black": black.name, "Result": result, "Termination": reason})
    return game, result, reason, len(board.move_stack), t_white, t_black


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    naluri, laya_ = Naluri(), Laya()
    out = Path("results/chess"); out.mkdir(parents=True, exist_ok=True)
    score = {naluri.name: 0.0, laya_.name: 0.0}
    for g in range(1, args.games + 1):
        white, black = (naluri, laya_) if g % 2 else (laya_, naluri)
        game, result, reason, plies, tw, tb = play(white, black, args.max_plies, rng, g)
        pts = {"1-0": (1, 0), "0-1": (0, 1)}.get(result, (0.5, 0.5))
        score[white.name] += pts[0]; score[black.name] += pts[1]
        (out / f"game{g}.pgn").write_text(str(game) + "\n")
        print(f"game {g}: {white.name} (white) vs {black.name} (black) -> {result} | {reason} | {plies} plies | "
              f"think time white {tw / max(1, (plies + 1) // 2) * 1000:.0f} ms/move, black {tb / max(1, plies // 2) * 1000:.0f} ms/move")
    print("final score:", score)


class Random:
    name = "Random mover"

    def scores(self, state, question, options):
        return {k: random.random() for k in options}
