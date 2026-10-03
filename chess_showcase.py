"""Play a logged showcase game (Naluri white vs Laya black) for the replay viewer.

Same move selection as chess_battle.py, but every move records what the model saw in its final round: the chosen
move, its probability and the top alternatives. Plays seeds until a game ends in checkmate (or --tries runs out)
and writes results/chess/showcase.json.
"""
import argparse
import json
import random
import time

import chess

import chess_battle as cb


def choose_logged(player, board, rng):
    moves = list(board.legal_moves)
    rng.shuffle(moves)
    san = {board.san(m): m for m in moves}
    desc = {board.san(m): cb.describe(board, m) for m in moves}
    state = cb.state_of(board)
    question = f"Which move is best for {state['side_to_move']} in this chess position?"
    pool, rounds = list(san), 0
    while len(pool) > cb.GROUP:
        winners = []
        for i in range(0, len(pool), cb.GROUP):
            group = pool[i:i + cb.GROUP]
            if len(group) == 1:
                winners += group
                continue
            p = player.scores(state, question, {k: desc[k] for k in group})
            winners.append(max(p, key=p.get))
        pool, rounds = winners, rounds + 1
    if len(pool) == 1:
        probs = {pool[0]: 1.0}
    else:
        probs = player.scores(state, question, {k: desc[k] for k in pool})
    ranked = sorted(probs, key=probs.get, reverse=True)
    best = ranked[0]
    return san[best], {
        "san": best, "desc": desc[best], "prob": round(float(probs[best]), 3), "legal": len(moves), "rounds": rounds,
        "alternatives": [{"san": k, "desc": desc[k], "prob": round(float(probs[k]), 3)} for k in ranked[1:4]],
    }


def play_logged(white, black, max_plies, rng):
    board, log = chess.Board(), []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        player = white if board.turn else black
        color = "white" if board.turn else "black"
        t = time.perf_counter()
        move, info = choose_logged(player, board, rng)
        info.update({"ply": len(board.move_stack) + 1, "color": color, "player": player.name, "uci": move.uci(),
                     "ms": round((time.perf_counter() - t) * 1000)})
        board.push(move)
        info.update({"fen": board.fen(), "check": board.is_check(),
                     "material": {"white": cb.material(board, chess.WHITE), "black": cb.material(board, chess.BLACK)}})
        log.append(info)
    if board.is_game_over(claim_draw=True):
        result, reason = board.result(claim_draw=True), board.outcome(claim_draw=True).termination.name.lower()
    else:
        mw, mb = cb.material(board, chess.WHITE), cb.material(board, chess.BLACK)
        result = "1-0" if mw > mb + 1 else "0-1" if mb > mw + 1 else "1/2-1/2"
        reason = "adjudicated on material"
    return log, result, reason


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--max-plies", type=int, default=120)
    args = ap.parse_args()
    naluri, laya_ = cb.Naluri(), cb.Laya()
    best = None
    for seed in range(args.tries):
        log, result, reason = play_logged(naluri, laya_, args.max_plies, random.Random(100 + seed))
        print(f"seed {seed}: {result} ({reason}, {len(log)} plies)")
        if reason == "checkmate":
            best = (seed, log, result, reason)
            break
        if best is None:
            best = (seed, log, result, reason)
    seed, log, result, reason = best
    data = {"white": naluri.name, "black": laya_.name, "result": result, "termination": reason, "seed": seed,
            "start_fen": chess.STARTING_FEN, "moves": log}
    json.dump(data, open("results/chess/showcase.json", "w"), indent=1, ensure_ascii=False)
    print(f"showcase: seed {seed}, {result} by {reason}, {len(log)} plies -> results/chess/showcase.json")
