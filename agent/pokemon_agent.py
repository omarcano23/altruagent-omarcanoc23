from altruagent import GameState, LegalAction
from examples.smoke_agent import choose_action as smoke_choose_action


# Match-specific memory.
#
# Each tournament/test session gets its own state so concurrent
# matches cannot interfere with each other.
#
# Structure:
# {
#     session_id: {
#         "active_by_slot": {
#             0: "pokemon_species",
#             1: "pokemon_species",
#         },
#         "last_switch_turn": {
#             "pokemon_species": turn_number,
#         },
#     }
# }
_session_memory: dict[str, dict] = {}


def choose_action(
    state: GameState,
    context,
) -> LegalAction | dict:
    """
    Deterministic Pokémon agent.

    Current baseline:
    - Smoke agent for draft and team preview.
    - During battle:
        * filter moves with zero PP
        * allow Fake Out only on the first turn after entering
        * target the opponent with the lowest HP fraction
    - Fall back to the tested smoke agent if anything unexpected happens.
    """

    if state.phase != "moving":
        return smoke_choose_action(state, context)

    observation = state.raw.get(
        "observation",
        {},
    )

    try:
        update_session_memory(
            state,
            observation,
        )

        return choose_doubles_turn(
            state,
            observation,
        )

    except Exception as exc:
        print(
            f"[pokemon_agent] fallback: "
            f"{type(exc).__name__}: {exc}"
        )

        return smoke_choose_action(
            state,
            context,
        )


def update_session_memory(
    state: GameState,
    observation: dict,
) -> None:
    """
    Track which Pokémon occupies each active slot.

    A Pokémon is considered to have just entered the field
    when the species in a slot changes from the previous
    observation.

    This avoids relying on the accumulated protocol log.
    """

    session_id = state.session_id

    memory = _session_memory.setdefault(
        session_id,
        {
            "active_by_slot": {},
            "last_switch_turn": {},
        },
    )

    current_turn = observation.get("turn")

    if current_turn is None:
        return

    active = observation.get(
        "active_pokemon",
        [],
    )

    for slot_number in (0, 1):

        pokemon = (
            active[slot_number]
            if slot_number < len(active)
            else None
        )

        species = None

        if pokemon and pokemon.get("species"):
            species = normalize_species(
                pokemon["species"]
            )

        previous_species = memory[
            "active_by_slot"
        ].get(slot_number)

        # A new Pokémon has entered this slot.
        #
        # This also handles the initial battle state:
        # previous_species is None, so the starting Pokémon
        # is considered to have entered on turn 1.
        if (
            species is not None
            and species != previous_species
        ):
            memory["last_switch_turn"][species] = (
                current_turn
            )

        memory["active_by_slot"][slot_number] = species


def normalize_species(
    species: str,
) -> str:
    """
    Normalize species names so that equivalent representations
    can be compared reliably.
    """

    return (
        species
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("'", "")
    )


def choose_doubles_turn(
    state: GameState,
    observation: dict,
) -> dict:
    """
    Build a legal doubles-turn action from the server-provided
    action template.
    """

    action = next(
        action
        for action in state.legal_actions
        if action.action_id == "doubles_turn"
    )

    template = action.input["action"]

    slots = sorted(
        template["slots"],
        key=lambda slot: slot.get("slot", 0),
    )

    choices = []

    for slot in slots:

        options = slot["options"]

        option = choose_best_option(
            options,
            slot,
            state,
        )

        choices.append(
            build_option(
                option,
                observation,
            )
        )

    print(
        f"[pokemon_agent] turn decision: "
        f"slot_0={choices[0]} | "
        f"slot_1={choices[1]}"
    )

    return {
        "type": "doubles_turn",
        "slot_0": choices[0],
        "slot_1": choices[1],
    }


def choose_best_option(
    options: list[dict],
    slot: dict,
    state: GameState,
) -> dict:
    """
    Choose the first option that passes our current
    action filters.

    We intentionally do not score moves yet.
    """

    observation = state.raw.get(
        "observation",
        {},
    )

    active = get_active_pokemon(
        observation,
        slot.get("slot", 0),
    )

    for option in options:

        if not option_is_usable(
            option,
            active,
            state,
        ):
            continue

        return option

    # If all options were filtered out, preserve
    # server-provided legality.
    return options[0]


def option_is_usable(
    option: dict,
    active: dict | None,
    state: GameState,
) -> bool:
    """
    Current action filters:

    1. Reject moves with zero PP.
    2. Reject Fake Out unless the Pokémon just entered
       the field this turn.
    """

    if option["type"] != "move":
        return True

    current_pp = option.get("current_pp")

    if (
        current_pp is not None
        and current_pp <= 0
    ):
        return False

    move_id = option.get("move_id")

    if move_id == "fakeout":

        return fake_out_is_available(
            active,
            state,
        )

    return True


def fake_out_is_available(
    active: dict | None,
    state: GameState,
) -> bool:
    """
    Fake Out only works on the first turn after
    the Pokémon enters the field.
    """

    if not active:
        return False

    species = active.get("species")

    if not species:
        return False

    observation = state.raw.get(
        "observation",
        {},
    )

    current_turn = observation.get("turn")

    if current_turn is None:
        return False

    memory = _session_memory.get(
        state.session_id,
        {},
    )

    last_switch_turn = memory.get(
        "last_switch_turn",
        {},
    )

    normalized_species = normalize_species(
        species
    )

    switch_turn = last_switch_turn.get(
        normalized_species
    )

    return switch_turn == current_turn


def build_option(
    option: dict,
    observation: dict,
) -> dict:
    """
    Convert a server option into the actual action payload.
    """

    if option["type"] == "pass":
        return {
            "type": "pass",
        }

    if option["type"] == "switch":
        return {
            "type": "switch",
            "species": option["species"],
        }

    choice = {
        "type": "move",
        "move_id": option["move_id"],
    }

    targets = option.get("targets") or []

    if targets:
        choice["target"] = choose_best_target(
            targets,
            observation,
        )

    return choice


def choose_best_target(
    targets: list[int],
    observation: dict,
) -> int:
    """
    Current target heuristic:

    Choose the legal opponent target with the lowest
    HP fraction.

    Target convention:
        1, 2 = opponent slots
       -1, -2 = ally slots
    """

    opponent_active = observation.get(
        "opponent_active_pokemon",
        [],
    )

    opponent_hp = {}

    for index, pokemon in enumerate(
        opponent_active
    ):

        if not pokemon:
            continue

        hp_fraction = pokemon.get(
            "current_hp_fraction"
        )

        if hp_fraction is None:

            current_hp = pokemon.get(
                "current_hp"
            )

            max_hp = pokemon.get(
                "max_hp"
            )

            if (
                current_hp is not None
                and max_hp
            ):
                hp_fraction = (
                    current_hp / max_hp
                )

        if hp_fraction is not None:
            opponent_hp[index + 1] = (
                hp_fraction
            )

    opponent_targets = [
        target
        for target in targets
        if target in opponent_hp
    ]

    if opponent_targets:

        return min(
            opponent_targets,
            key=lambda target: opponent_hp[target],
        )

    # For ally/support targets, or any unusual target
    # configuration, preserve the first server-provided
    # legal target.
    return targets[0]


def get_active_pokemon(
    observation: dict,
    slot_number: int,
) -> dict | None:
    """
    Return the Pokémon currently occupying a battle slot.
    """

    active = observation.get(
        "active_pokemon",
        [],
    )

    if slot_number >= len(active):
        return None

    return active[slot_number]


def create_agent():
    return choose_action