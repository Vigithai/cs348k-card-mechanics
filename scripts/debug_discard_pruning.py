"""Print flush-biased discard pruning details for a specific hand."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import analyze_pruned_discard_candidates, create_standard_deck


SAMPLE_HAND = ["AH", "KH", "7H", "3H", "9C", "9S", "2D"]
SUIT_CODE_TO_NAME = {
    "C": "club",
    "D": "diamond",
    "H": "heart",
    "S": "spade",
}


def parse_card_codes(card_codes: list[str]) -> list:
    """Resolve compact card codes like AH or 10D into deck Card objects."""
    deck_lookup = {
        (card.rank, card.suit): card
        for card in create_standard_deck()
    }

    resolved_cards = []
    for card_code in card_codes:
        normalized_code = card_code.strip().upper()
        if len(normalized_code) < 2:
            raise ValueError(f"Invalid card code: {card_code!r}")

        suit_code = normalized_code[-1]
        rank_code = normalized_code[:-1]
        rank = "10" if rank_code == "T" else rank_code
        if suit_code not in SUIT_CODE_TO_NAME:
            raise ValueError(f"Unsupported suit code in {card_code!r}")

        suit = SUIT_CODE_TO_NAME[suit_code]
        try:
            resolved_cards.append(deck_lookup[(rank, suit)])
        except KeyError as error:
            raise ValueError(f"Unsupported card code: {card_code!r}") from error
    return resolved_cards


def format_card(card) -> str:
    """Return a short human-readable card label."""
    return f"{card.rank} of {card.suit}"


def main() -> None:
    """Parse a hand and print pruning details for manual inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cards",
        nargs="*",
        help="Hand cards like AH KH 7H 3H 9C 9S 2D. Defaults to a sample flush-heavy hand.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=5,
        help="How many of the most discardable cards to seed the subset search with.",
    )
    args = parser.parse_args()

    hand = parse_card_codes(args.cards or SAMPLE_HAND)
    analysis = analyze_pruned_discard_candidates(hand, candidate_pool_size=args.candidate_pool_size)

    print("Hand:")
    for index, card in enumerate(hand):
        print(
            f"  [{index}] {format_card(card):16s} "
            f"chip={card.chip_value:2d} keep={analysis.keep_scores_by_index[index]:5.2f}"
        )

    print("\nBest Current Play:")
    print(
        f"  category={analysis.best_play_category} "
        f"score={analysis.best_play_score} "
        f"action={analysis.best_play_action}"
    )
    print(f"  four_of_a_kind_short_circuited={analysis.four_of_a_kind_short_circuited}")

    print("\nRanked Discardability:")
    for rank, index in enumerate(analysis.ranked_discardable_indices, start=1):
        card = hand[index]
        print(
            f"  {rank}. [{index}] {format_card(card):16s} "
            f"keep={analysis.keep_scores_by_index[index]:5.2f}"
        )

    print("\nPruned Discard Actions:")
    if not analysis.pruned_discard_actions:
        print("  <none>")
    else:
        for action in analysis.pruned_discard_actions:
            print(f"  {action}")


if __name__ == "__main__":
    main()
