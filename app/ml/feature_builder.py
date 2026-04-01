from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session


MODE_NUMERIC_FEATURES: dict[str, list[str]] = {
    "pathOfLegend": ["evolution_count", "league_number"],
    "ladder": ["avg_level", "evolution_count", "starting_trophies"],
}
DEFAULT_NUMERIC = ["evolution_count", "league_number"]


class FeatureBuilder:
    def __init__(self, api_id_to_index: dict[int, int], evo_card_to_index: dict[int, int]):
        self.api_id_to_index = api_id_to_index
        self.evo_card_to_index = evo_card_to_index
        self._num_cards = len(api_id_to_index)
        self._num_evo = len(evo_card_to_index)
        self._deck_size = self._num_cards + self._num_evo

    @classmethod
    def from_card_rows(cls, rows: list[tuple[int, str, int | None]]) -> FeatureBuilder:
        unique_ids: set[int] = set()
        evo_ids: set[int] = set()
        for api_id, card_type, _ in rows:
            unique_ids.add(api_id)
            if card_type == "EVOLUTION":
                evo_ids.add(api_id)
        sorted_ids = sorted(unique_ids)
        sorted_evo = sorted(evo_ids)
        return cls(
            {aid: i for i, aid in enumerate(sorted_ids)},
            {aid: i for i, aid in enumerate(sorted_evo)},
        )

    @classmethod
    def from_db(cls, session: Session) -> FeatureBuilder:
        result = session.execute(
            text("SELECT DISTINCT api_id, card_type, max_evo_level FROM cards")
        )
        return cls.from_card_rows([(r[0], r[1], r[2]) for r in result])

    def _numeric_features(self, battle_type: str) -> list[str]:
        return MODE_NUMERIC_FEATURES.get(battle_type, DEFAULT_NUMERIC)

    def feature_names(self, battle_type: str) -> list[str]:
        numeric = self._numeric_features(battle_type)
        sorted_ids = sorted(self.api_id_to_index, key=self.api_id_to_index.get)
        sorted_evo = sorted(self.evo_card_to_index, key=self.evo_card_to_index.get)
        names = []
        for prefix in ("my", "opp"):
            for aid in sorted_ids:
                names.append(f"{prefix}_card_{aid}")
            for aid in sorted_evo:
                names.append(f"{prefix}_evo_{aid}")
            for feat in numeric:
                names.append(f"{prefix}_{feat}")
        return names

    def encode_deck(self, card_ids: list[int], card_levels: list[int], evo_levels: list[int]) -> np.ndarray:
        vec = np.zeros(self._deck_size, dtype=np.float32)
        for cid, lvl, evo in zip(card_ids, card_levels, evo_levels):
            idx = self.api_id_to_index.get(cid)
            if idx is not None:
                vec[idx] = float(lvl)
            if evo > 0 and cid in self.evo_card_to_index:
                vec[self._num_cards + self.evo_card_to_index[cid]] = 1.0
        return vec

    def _numeric_vector(self, battle_type: str, avg_level: float, evolution_count: int,
                        league_number: int | None, starting_trophies: int | None) -> np.ndarray:
        values = {
            "avg_level": float(avg_level),
            "evolution_count": float(evolution_count),
            "league_number": float(league_number) if league_number is not None else -1.0,
            "starting_trophies": float(starting_trophies) if starting_trophies is not None else -1.0,
        }
        return np.array([values[f] for f in self._numeric_features(battle_type)], dtype=np.float32)

    def build_matchup_vector(self, battle_type: str,
                             deck_card_ids: list[int], deck_card_levels: list[int], deck_evo_levels: list[int],
                             opponent_card_ids: list[int], opponent_card_levels: list[int], opponent_evo_levels: list[int],
                             avg_level: float, evolution_count: int,
                             league_number: int | None, starting_trophies: int | None) -> np.ndarray:
        my_side = np.concatenate([
            self.encode_deck(deck_card_ids, deck_card_levels, deck_evo_levels),
            self._numeric_vector(battle_type, avg_level, evolution_count, league_number, starting_trophies),
        ])
        opp_side = np.concatenate([
            self.encode_deck(opponent_card_ids, opponent_card_levels, opponent_evo_levels),
            self._numeric_vector(battle_type, 0.0, 0, league_number, starting_trophies),
        ])
        return np.concatenate([my_side, opp_side])
