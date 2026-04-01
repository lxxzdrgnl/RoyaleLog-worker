import numpy as np
import pytest
from app.ml.feature_builder import FeatureBuilder


@pytest.fixture
def card_rows():
    return [
        (26000000, "NORMAL", None),
        (26000000, "EVOLUTION", 1),  # Knight has evolution
        (26000001, "NORMAL", None),
        (26000002, "NORMAL", None),
        (26000003, "NORMAL", None),
        (26000004, "NORMAL", None),
        (26000005, "NORMAL", None),
        (26000006, "NORMAL", None),
        (26000007, "NORMAL", None),
        (159000000, "NORMAL", None),
    ]


@pytest.fixture
def builder(card_rows):
    return FeatureBuilder.from_card_rows(card_rows)


def test_pol_feature_count(builder):
    names = builder.feature_names("pathOfLegend")
    unique_api_ids = 9
    evo_cards = 1
    numeric = 2  # evolution_count, league_number
    expected = (unique_api_ids + evo_cards + numeric) * 2
    assert len(names) == expected


def test_ladder_feature_count(builder):
    names = builder.feature_names("ladder")
    unique_api_ids = 9
    evo_cards = 1
    numeric = 3  # avg_level, evolution_count, starting_trophies
    expected = (unique_api_ids + evo_cards + numeric) * 2
    assert len(names) == expected


def test_encode_deck_level_and_evo(builder):
    card_ids    = [26000000, 26000001, 26000002, 26000003,
                   26000004, 26000005, 26000006, 159000000]
    card_levels = [13,       11,       12,       10,
                    9,        8,        7,        14]
    evo_levels  = [1,         0,        0,        0,
                   0,         0,        0,        0]
    vec = builder.encode_deck(card_ids, card_levels, evo_levels)

    knight_idx = builder.api_id_to_index[26000000]
    assert vec[knight_idx] == 13.0  # level value, not 1

    evo_start = len(builder.api_id_to_index)
    evo_knight_idx = evo_start + builder.evo_card_to_index[26000000]
    assert vec[evo_knight_idx] == 1.0

    arrows_idx = builder.api_id_to_index[26000007]
    assert vec[arrows_idx] == 0.0  # not in deck


def test_build_matchup_vector_pol(builder):
    my_cards  = [26000000, 26000001, 26000002, 26000003,
                 26000004, 26000005, 26000006, 159000000]
    opp_cards = [26000001, 26000002, 26000003, 26000004,
                 26000005, 26000006, 26000007, 159000000]
    vec = builder.build_matchup_vector(
        battle_type="pathOfLegend",
        deck_card_ids=my_cards, deck_card_levels=[13]*8, deck_evo_levels=[1,0,0,0,0,0,0,0],
        opponent_card_ids=opp_cards, opponent_card_levels=[12]*8, opponent_evo_levels=[0]*8,
        avg_level=13.0, evolution_count=1, league_number=7, starting_trophies=None,
    )
    names = builder.feature_names("pathOfLegend")
    assert len(vec) == len(names)
    assert "my_avg_level" not in names
    assert "my_league_number" in names


def test_build_matchup_vector_ladder(builder):
    my_cards  = [26000000, 26000001, 26000002, 26000003,
                 26000004, 26000005, 26000006, 159000000]
    opp_cards = [26000001, 26000002, 26000003, 26000004,
                 26000005, 26000006, 26000007, 159000000]
    vec = builder.build_matchup_vector(
        battle_type="ladder",
        deck_card_ids=my_cards, deck_card_levels=[13]*8, deck_evo_levels=[0]*8,
        opponent_card_ids=opp_cards, opponent_card_levels=[12]*8, opponent_evo_levels=[0]*8,
        avg_level=13.0, evolution_count=0, league_number=None, starting_trophies=5500,
    )
    names = builder.feature_names("ladder")
    assert len(vec) == len(names)
    assert "my_avg_level" in names
    assert "my_starting_trophies" in names
